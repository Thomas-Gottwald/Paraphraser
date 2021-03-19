import torch
import spacy
import re
import numpy as np
import pandas as pd
import random
from transformers import AutoConfig, AutoTokenizer, AutoModelForMaskedLM, AutoModelForCausalLM

from getPath import get_local_path
import os
from datetime import datetime
from typing import Optional
from tqdm import tqdm

def init_model(model_name_or_path: str, max_len: int, enable_cuda: bool=True):
    config = AutoConfig.from_pretrained(model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, model_max_length=max_len)
    torch_device = 'cuda' if enable_cuda and torch.cuda.is_available() else 'cpu'
    if model_name_or_path == 'xlnet-large-cased':
        masked_model = AutoModelForCausalLM.from_pretrained(model_name_or_path, config=config).to(torch_device)
    else:
        masked_model = AutoModelForMaskedLM.from_pretrained(model_name_or_path, config=config).to(torch_device)

    return tokenizer, masked_model

def check_token(token: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z]+([-'][A-Za-z]+)*", token))

def spin_text(text: str, tokenizer, model, mask_prob: float, max_prob: float=0.1, k: int=5, seed: Optional[int]=None) -> (str, pd.DataFrame):
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
    # shift of the index for texts split intu multiple paragraphs
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
        max_tokens = int(len_tokens * max_prob)
        # chose the tokens that should be replaced (so that at most max_tokens get replaced at once)
        mask_idc = rng.choice(indices, size=(max(N//max_tokens, 1),min(max_tokens, N)), replace=False)
        if N > max_tokens and N % max_tokens != 0:
            rest_idc = np.setdiff1d(indices, mask_idc)
            if N < len(indices):
                rest_idc = rng.choice(rest_idc, size=N%max_tokens, replace=False)
            mask_idc = np.append(mask_idc, np.append(rest_idc, (max_tokens-N%max_tokens)*[-1]).reshape(1,-1), axis=0)
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

def create_sample(sample_size: int, data: list, spin_text_args: list, disguise_sample: bool=False):
    # set up the language model
    model_name = 'roberta-large'
    max_seq_len = 512
    tokenizer, lm = init_model(model_name, max_seq_len)
    # the name of the folder to stor this sample
    sampleFolder = datetime.now().strftime("%Y-%m-%d %H_%M_%S")
    # set the paths to the data
    path = get_local_path()
    data_paths = []
    for d in data:
        data_paths.append(os.path.join(path, *['data', d]))
    # get the original text file names
    ogfiles = []
    for dp in data_paths:
        ogfiles.append([f for f in os.listdir(os.path.join(dp, 'ogUTF-8')) if os.path.isfile(os.path.join(dp, *['ogUTF-8', f]))])
    # chose the sample files
    sample_files = set()
    for i in range(sample_size):
        chose_data = random.randint(0, 10 * len(data)-1) // 10
        chose_file = random.sample(ogfiles[chose_data], k=1)[0]
        sample_files.add((data[chose_data], chose_file))
    # creating the folder for the sample if not exits
    if not os.path.exists(os.path.join(path, *['data', 'sample', sampleFolder])):
        os.makedirs(os.path.join(path, *['data', 'sample', sampleFolder]))

    if disguise_sample:
        # stores the information in witch order the text occur in the sample files (for disguise_sample=True)
        logInfo = []
    # write the samples
    for d, sf in tqdm(sample_files):
        # open the sample file
        sfile_name = sf.replace('ORIG', d.upper())
        sfile = open(os.path.join(path, *['data', 'sample', sampleFolder, sfile_name]), 'w', encoding='utf-8', newline='\n')
        # create DataFrame file
        dffile_name = 'df_' + sfile_name
        dffile = open(os.path.join(path, *['data', 'sample', sampleFolder, dffile_name]), 'w', encoding='utf-8', newline='\n')
        # the order in which the different spun_texts and the original text (index -1) are handeld
        order = [i for i in range(-1, len(spin_text_args))]

        if disguise_sample:
            # information in witch order the text occur this sample file
            sfileInfo = [sfile_name]
            # shuffle the oder (with -1 for the original text)
            random.shuffle(order)

        for o in order:
            if disguise_sample:
                sfileInfo.append(o)# remember the order
            else:
                if o == -1:
                    arg_info_text = "Original Text"
                else:
                    arg_info_text = ", ".join(['{}={}'.format(arg, spin_text_args[o][arg]) for arg in spin_text_args[o]])
                sfile.write(arg_info_text + ": \n\n")

            og_path = os.path.join(data_paths[data.index(d)], *['ogUTF-8', sf])
            with open(og_path, encoding='utf-8') as file:
                originalText = file.read()
            if o == -1:
                sfile.write(originalText)
            else:
                spun_text, df = spin_text(originalText, tokenizer, lm, **spin_text_args[o])
                sfile.write(spun_text)
                # Write in the DataFrame file
                arg_info_text = ", ".join(['{}={}'.format(arg, spin_text_args[o][arg]) for arg in spin_text_args[o]])
                dffile.write(arg_info_text + ": \n\n")
                dffile.write(df.sort_values(
                    ['index', 'top-k'],
                    key=lambda series : series.astype(int) if series.name == 'index' else series
                ).to_string())
                dffile.write('\n\n' + 100*'-' + '\n\n')

            sfile.write('\n\n' + 100*'-' + '\n\n')

        if disguise_sample:
            logInfo.append(sfileInfo)

        # close the sample file and DataFrame file
        sfile.close()
        dffile.close()

    if disguise_sample:
        # create the Info file where the information for the sample is stored
        with open(os.path.join(path, *['data', 'sample', sampleFolder, 'Info.txt']), 'w', encoding='utf-8', newline='\n') as file:
            for info in logInfo:
                file.write(info[0] + '\n')
                for i in range(1, len(info)):
                    if info[i] == -1:
                        file.write("    original Text\n")
                    else:
                        file.write("    " + ", ".join(['{}={}'.format(arg, spin_text_args[info[i]][arg]) for arg in spin_text_args[info[i]]]) + "\n")
                file.write('\n')


if __name__ == '__main__':

    # seed = datetime.now().microsecond
    # create_sample(
    #     3,
    #     ['wikipedia'],
    #     [{'mask_prob': 0.5, 'max_prob': 0.1, 'k': 5, 'seed': seed}],
    #     disguise_sample=False
    # )
    # quit()

    model_name = 'xlnet-large-cased' # really bad for unmasking in genearal
    # model_name = 'facebook/bart-large'
    # model_name = 'roberta-large'
    max_seq_len = 512
    mask_prob = 0.5
    k = 5
    seed = datetime.now().microsecond
    tokenizer, lm = init_model(model_name, max_seq_len, enable_cuda=False)

    #232530-ORIG-13.txt
    #1208667-ORIG-4.txt
    #27509373-ORIG-28.txt
    path = os.path.join(get_local_path(), *['data', 'wikipedia', 'ogUTF-8', '4115833-ORIG-15.txt'])
    with open(path, 'r', encoding='utf-8') as file:
        toy_sentence = file.read()
    print('')
    print(toy_sentence)
    print('')

    spun_text, df = spin_text(toy_sentence, tokenizer, lm, mask_prob, k=k, seed=seed)

    print(spun_text)

    print(df)

    # with open(os.path.join(get_local_path(), *['data', 'test.txt']), 'w', encoding='utf-8', newline='\n') as file:
    #     file.write(df.sort_values(
    #         ['index', 'top-k'],
    #         key=lambda series : series.astype(int) if series.name == 'index' else series
    #     ).to_string())
