import torch
import spacy
import re
import numpy as np
import pandas as pd
import os
from transformers import AutoConfig, AutoTokenizer, AutoModelForMaskedLM
from enum import Enum
from getPath import get_local_path
from datetime import datetime
from typing import Optional, Union

class Data(Enum):
    """
    Enum for the three data sets
    wikipedia, arxiv and thesis
    """
    THESIS = 1
    ARXIV = 2
    WIKIPEDIA = 3

    def __str__(self):
        if self is Data.THESIS:
            return 'thesis'
        elif self is Data.ARXIV:
            return 'arxiv'
        else:
            return 'wikipedia'

class Model(Enum):
    """
    Enum for the mask language models
    
        ROBERTA: 'roberta-large'
        BART: 'facebook/bart-large'
    """
    ROBERTA = 1
    BART = 2

    def __str__(self):
        if self is Model.BART:
            return 'facebook/bart-large'
        else:
            return 'roberta-large'

def init_model(model_type: Union[Model,str], max_len: int, enable_cuda: bool=True):
    """
    Initialize the neural language model and its tokenizer

    Args:
        model_type: Enum for the mask neural language model
            or a string referring to a mask language model that can
            be loaded by AutoModelForMaskLM from transformers
        max_len: The maximum input length of the model
        enable_cuda: Wether cuda should be enabled or not

    Return:
        tokenizer, masked_model: The model tokenizer and the masked language model
    """
    config = AutoConfig.from_pretrained(str(model_type))
    tokenizer = AutoTokenizer.from_pretrained(str(model_type), model_max_length=max_len)
    torch_device = 'cuda' if enable_cuda and torch.cuda.is_available() else 'cpu'
    masked_model = AutoModelForMaskedLM.from_pretrained(str(model_type), config=config).to(torch_device)

    return tokenizer, masked_model

def check_token(token: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z]+([-'][A-Za-z]+)*", token))

def spin_text(text: str, tokenizer, model, mask_prob: float, max_prob: float=0.1, k: int=5, seed: Optional[int]=None) -> (str, pd.DataFrame):
    """
    Paraphrases the input text by replacing some of its words

    Args:
        text: The text to be paraphrased
        tokenizer: The tokenizer of a masking model
        model: A masking language model
        mask_prob: The amount of works to be masked
        max_prob: The maximum amount of words that can be replaced at once
        k: How many suggested tokens should be concidered as new tokens
        seed: A seed for choosing which tokens should be replaced

    Return:
        spun_text, df: Tha spun paragraph and a DataFrame containing for each replaced word
        the index, sting and POS of the original token and POS and score of the top-k suggested tokens
    """
    # split the paragraph into sentences which fit into the model
    # set up the sentencizer
    nlp_sentencizer = spacy.load("en_core_web_sm", disable=["parser", "tagger", "ner"])
    nlp_sentencizer.add_pipe(spacy.lang.en.English().create_pipe("sentencizer"))
    work_text = text
    paragraphs = []
    while work_text != '':
        tokens = tokenizer(
            work_text,
            max_length=tokenizer.model_max_length,
            truncation=True
        )
        text_decoded = tokenizer.decode(tokens['input_ids'])[3:-4]# slicing excludes the start and end tokens (adjust for other models)
        # when the work text is to long for the model the decoded text is shorter
        if text_decoded == work_text or len(tokens['input_ids']) < tokenizer.model_max_length:
            paragraphs.append(work_text)
            work_text = ''
        else:
            work_sents = list(nlp_sentencizer(text_decoded).sents)
            # to not have an endles while loop
            assert work_sents[-1].start_char > 0, "The original text contains a sentence to longer for the model input!"
            paragraphs.append(work_text[:work_sents[-1].start_char])
            work_text = work_text[work_sents[-1].start_char:]
    # spin each paragraph
    nlp = spacy.load("en_core_web_sm", disable=["parser"])
    rng = np.random.default_rng(seed)
    # list for the spun paragraphs
    spun_paragraphs = []
    # DataFrame is set later
    df = None
    # shift of the index for texts split into multiple paragraphs
    df_index_shift = 0
    for paragraph in paragraphs:
        # fullword tokenization with POS tanging and name entity recognition
        doc = nlp(paragraph)
        # indices of tokens in the fullword tokenization that can get masked
        indices = np.array([i for i in range(len(doc))])
        for ent in doc.ents:
            indices[ent.start:ent.end] = -1
        indices = np.array([i for i in indices if i >= 0 and check_token(doc[i].text)])
        # "100% of words"
        len_tokens = len([1 for t in doc if check_token(t.text)])
        # how many tokens to mask
        N = min(max(int(len_tokens * mask_prob), 0), len(indices))
        # maximum number of tokens that can be masked at once
        max_tokens = max(int(len_tokens * max_prob), 1)
        # chose the tokens that should be replaced (so that at most max_tokens get replaced at once)
        mask_idc = rng.choice(indices, size=(max(N//max_tokens, 1),min(max_tokens, N)), replace=False)
        if N > max_tokens and N % max_tokens != 0:
            rest_idc = np.setdiff1d(indices, mask_idc)
            if N < len(indices):
                rest_idc = rng.choice(rest_idc, size=N%max_tokens, replace=False)
            mask_idc = np.append(mask_idc, np.append(rest_idc, (max_tokens-N%max_tokens)*[-1]).reshape(1,-1), axis=0)
        # make sure that tokens where chosen to be masked
        assert len(mask_idc[-1]) > 0, "Found no tokens to mask!"
        mask_idc = np.sort(mask_idc)
        # list for storing the resluting tokens
        memorize_results = []
        # go over all lists of mask indices
        for j, mask_list in enumerate(mask_idc):
            # remove leading negative ones from the mask list
            if mask_list[0] == -1:
                mask_list = mask_list[mask_list != -1]
            # insert the mask tokens in the input text
            input_text = paragraph
            for m_idx in mask_list[::-1]:
                input_text = input_text[:doc[m_idx].idx] + tokenizer.mask_token + input_text[doc[m_idx].idx+len(doc[m_idx]):]
            # model tokenization
            inputs = tokenizer(
                input_text,
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors='pt'
            ).to(model.device)
            # masking model
            preds = model(**inputs)
            # activate out put with softmax function
            activated_preds = torch.softmax(preds[0], dim=2)
            # filter out the predictions
            masked_token_indices = (inputs['input_ids'] == tokenizer.mask_token_id)
            masked_preds = activated_preds[masked_token_indices]
            # get the top k predictions
            topk = torch.topk(masked_preds, k, dim=1)
            # choosing new tokens
            results = torch.zeros((topk.indices.shape[0], 1), dtype=int).to(model.device)
            for i, m_idx in enumerate(mask_list):
                old_t = doc[m_idx].text.lower()# old token in lower cases to avoid replacements with not really new tokens
                check_t = []# list for the previous and next token to avoide repetition of tokens
                if m_idx > 0:
                    if m_idx-1 in mask_list:
                        # because mask_list is sorted if m_idx-1 is in mask_list it was processed in the last loop
                        # so in new_t is the previous token in the final text stored
                        check_t.append(new_t)
                    elif m_idx-1 in mask_idc[:j]:
                        # the resulting token strings are stored with the same index as in mask_idc
                        # therefor at this position is the previous token in the final text stored
                        m_r_I, m_r_J = np.where(mask_idc[:j]==m_idx-1)
                        check_t.append(memorize_results[m_r_I[0]][m_r_J[0]])
                    else:
                        check_t.append(doc[m_idx-1].text)# the previous token in the original text
                if m_idx+1 < len(doc):
                    if m_idx+1 in mask_idc[:j]:
                        # the resulting token strings are stored with the same index as in mask_idc
                        # therefor at this position is the next token in the final text stored
                        m_r_I, m_r_J = np.where(mask_idc[:j]==m_idx+1)
                        check_t.append(memorize_results[m_r_I[0]][m_r_J[0]])
                    else:
                        check_t.append(doc[m_idx+1].text)# the next token in the original text
                for pos in range(k):
                    new_t = tokenizer.decode(topk.indices[i,pos]).replace(' ', '')
                    if new_t.lower() != old_t and new_t not in check_t:
                        results[i] = pos
                        break
            # setting the restulting tokens and their score values
            results_indices = topk.indices.gather(1, results)
            # store the resluting token strings
            if N < max_tokens or len(mask_list) == max_tokens:
                memorize_results.append([tokenizer.decode(t) for t in results_indices.flatten()])
            else:
                # when mask_list contains place holder tokens
                # insert place holder at the same postitions in memorize_results
                memorize_results.append((max_tokens-len(mask_list))*[''])
                memorize_results[-1].extend([tokenizer.decode(t) for t in results_indices.flatten()])

            # create the DataFrame
            # add the mask indices
            df_np_array = np.tile(mask_list+df_index_shift, (k, 1)).transpose().reshape(1, -1)
            # add the original tokens
            df_np_array = np.append(df_np_array, np.tile([doc[i].text for i in mask_list], (k, 1)).transpose().reshape(1, -1), axis=0)
            # add the original POS tags
            df_np_array = np.append(df_np_array, np.tile([doc[i].pos_ for i in mask_list], (k, 1)).transpose().reshape(1, -1), axis=0)
            # add the position numbers of the suggested tokens
            df_np_array = np.append(df_np_array, np.tile(np.arange(1, k+1), len(mask_list)).reshape(1, -1), axis=0)
            # add the suggested tokens
            topk_str = np.array([tokenizer.decode(t) for t in topk.indices.flatten()])
            df_np_array = np.append(df_np_array, topk_str.reshape(1, -1), axis=0)
            # add the POS tags of the suggested tokens
            new_pos = np.zeros(len(topk_str), dtype=object)
            for l in range(k):
                # for all k positions
                # insert all top l tokens into the pargraph
                # and store the positions where the were interted
                topk_paragraph = paragraph
                topk_mask_positions = []
                shift = 0
                for i, idx in enumerate(mask_list):
                    t = doc[idx]
                    new_t = topk_str[i*k+l]# the i-th token at position l
                    if idx > 0 and doc[idx-1].text_with_ws[-1] == ' ':
                        new_t = new_t.replace(' ', '')
                    start = t.idx+shift
                    end = start+len(new_t)
                    topk_mask_positions.append((start, end))
                    topk_paragraph = topk_paragraph[:start] + new_t + topk_paragraph[start+len(t):]
                    shift += len(new_t) - len(t)
                # tokenize the paragraph and determen POS tags
                topk_doc = nlp(topk_paragraph)
                # find the new tokens in the tokenization and store there POS tags
                i = 0
                topk_idx = 0
                for start, end in topk_mask_positions:
                    for t in topk_doc[topk_idx:]:
                        if t.i < len(topk_doc)-1 and t.idx < end and topk_doc[t.i+1].idx >= end:
                            topk_idx = t.i
                            break
                        elif t.i == len(topk_doc)-1 and topk_doc[t.i-1].idx < start:
                            topk_idx = t.i
                            break
                    new_pos[i*k+l] = topk_doc[topk_idx].pos_# POS for i-th token at the l-th position
                    i += 1
            df_np_array = np.append(df_np_array, new_pos.reshape(1, -1), axis=0)
            # add the scores
            df_np_array = np.append(df_np_array, topk.values.flatten().detach().cpu().numpy().reshape(1, -1), axis=0)
            # add maker for the chosen tokens
            mark_chosen = np.zeros(0)
            for pos in results.flatten():
                tmp_mark = k*['']
                tmp_mark[pos] = 'x'
                mark_chosen = np.append(mark_chosen, tmp_mark)
            df_np_array = np.append(df_np_array, mark_chosen.reshape(1, -1), axis=0)
            # insert the data into the DataFrame
            df_MultiIndex = pd.MultiIndex.from_arrays(df_np_array[:4], names=['index', 'og_token', 'og_POS', 'top-k'])
            if type(df) is pd.DataFrame:
                df = df.append(pd.DataFrame(df_np_array[4:].transpose() ,index=df_MultiIndex, columns=['token', 'POS', 'score', 'chosen']))
            else:
                df = pd.DataFrame(df_np_array[4:].transpose() ,index=df_MultiIndex, columns=['token', 'POS', 'score', 'chosen'])

        # shift the indices for the DataFrame ot the next paragraph
        df_index_shift += len(doc)
        # comput the 2d indices to sort mask_idc
        argsort_mask_idc_2d = np.dstack(np.unravel_index(np.argsort(mask_idc.ravel()), mask_idc.shape))[0]
        # inset the chosen token stings into the paragraph
        spun_paragraph = paragraph
        for i, j in argsort_mask_idc_2d[::-1]:
            # jump out of for loop when encountering a place holder index(-1)
            if mask_idc[i, j] == -1:
                break
            t = doc[mask_idc[i, j]]
            new_t = memorize_results[i][j]
            if mask_idc[i, j] > 0 and doc[mask_idc[i, j]-1].text_with_ws[-1] == ' ':
                new_t = new_t.replace(' ', '')
            spun_paragraph = spun_paragraph[:t.idx] + new_t + spun_paragraph[t.idx+len(t):]
        # store the spun paragraph
        spun_paragraphs.append(spun_paragraph)
    # if an original paragraph has no leading space than remove leading spaces from the spun paragraph
    # and if the original paragraph has a leading space and the spun paragraph dose not
    # than inset a space at the start of the spun paragraph
    for i, p in enumerate(paragraphs):
        if p[0] != ' ' and spun_paragraphs[i][0] == ' ':
            spun_paragraphs[i] = spun_paragraphs[i][1:]
        elif p[0] == ' ' and  spun_paragraphs[i][0] != ' ':
            spun_paragraphs[i] = ' ' + spun_paragraphs[i]
    # join the spun paragraphs together
    spun_text = ''.join(spun_paragraphs)

    return spun_text, df


if __name__ == '__main__':
    # Example Code for spinning a paragraph of one of the datasets

    data = Data.WIKIPEDIA

    model_type = Model.ROBERTA
    max_seq_len = 512
    mask_prob = 0.5
    k = 5
    seed = datetime.now().microsecond
    tokenizer, lm = init_model(model_type, max_seq_len)

    path = os.path.join(get_local_path(), *['data', str(data), 'ogUTF-8', '4115833-ORIG-15.txt'])
    with open(path, 'r', encoding='utf-8') as file:
        toy_sentence = file.read()
    print('')
    print(toy_sentence)
    print('')

    try:
        spun_text, df = spin_text(toy_sentence, tokenizer, lm, mask_prob, k=k, seed=seed)

        print(spun_text)
        print(df)
    except AssertionError as exc:
        print('AssertionError in spin_text: {}'.format(exc))
