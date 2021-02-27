import torch
import spacy
import re
import numpy as np
import pandas as pd
import random
import copy
from transformers import AutoConfig, AutoTokenizer, AutoModelForMaskedLM

from getPath import get_local_path
import os
from datetime import datetime
from typing import Optional

def init_model(model_name_or_path: str, max_len: int):
    config = AutoConfig.from_pretrained(model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, model_max_length=max_len)
    masked_model = AutoModelForMaskedLM.from_pretrained(model_name_or_path, config=config)

    return tokenizer, masked_model

def check_token(token: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z]+([-'][A-Za-z]+)*", token))

def tokenization_mapping(model_tok : list, doc) -> np.ndarray:
    # getting the indices of all named entities
    ne_list = []
    for ent in doc.ents:
        ne_list.extend([i for i in range(ent.start, ent.end)])

    mapping = []
    j = 0# doc index
    start = 0# start index for next token in model_tok
    token = ''
    for i, t in enumerate(model_tok):
        t = t.replace(' ', '')# remove leading spaces from t

        # while next doc token can be ingnorred (is name entity or has wrong form)
        while j in ne_list or not check_token(doc[j].text):
            j += 1
            if j >= len(doc):
                return np.array(mapping)

        # if t is next part of doc[j].text
        if doc[j].text.startswith(t, len(token), len(token)+len(t)):
            # if t is the first part of doc[j].text (then len(token)=0 => token='')
            if token == '':
                start = i
                token = t
            else:
                token += t
        else:
            # if not set token again to the empty string
            token = ''

        # if all parts of doc[j].text were found
        if token == doc[j].text:
            mapping.append([j, start, i+1])
            j += 1
            if j >= len(doc):
                break
            token = ''

    return np.array(mapping)

def mask_tokens_mapping(inputs: torch.Tensor, mapping: list, tokenizer, mask_prob: float) -> (torch.Tensor, np.ndarray):
    N = int(len(mapping)*mask_prob)
    # sample the tokens which should be masked
    mask_map = np.array(random.sample(mapping, k=N))

    masked_indices = torch.zeros(inputs['input_ids'].shape, dtype=bool)
    for _, a, b in mask_map:
        for i in range(a, b):
            masked_indices[0,i] = True
    # overide all subword tokens coresponding to a samled token with the mask token 
    inputs['input_ids'][masked_indices] = tokenizer.mask_token_id

    return inputs, mask_map


def spin_text_simple(paragraph: str, tokenizer, model, mask_prob: float, k: int=5) -> (str, pd.DataFrame):
    nlp = spacy.load("en_core_web_sm", disable=["parser"])
    # input gets cut off when longer as max_length 
    inputs = tokenizer(
        paragraph,
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors='pt'
    )
    # subword tokenization of the model tokenizer
    model_tok = [tokenizer.decode(t) for t in inputs['input_ids'].flatten()]
    # fullword tokenization with POS tanging and name entity recognition
    doc = nlp(paragraph)
    # mapping betwen the two tokenizations
    mapping = tokenization_mapping(model_tok, doc)
    # intert mask tokens and tokenization mapping of the masked tokens
    inputs, mask_map = mask_tokens_mapping(inputs, mapping, tokenizer, mask_prob)
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
    results = torch.zeros((topk.indices.shape[0], 1), dtype=int)
    shift = 0
    for i, m in enumerate(np.sort(mask_map, axis=0)):
        _, a, b = m
        # for all predicted subword token chose the first that is different form the old token
        for l in range(b - a):
            old_t = model_tok[a+l].replace(' ', '').lower()
            for pos in range(k):
                new_t = tokenizer.decode(topk.indices[i+shift+l,pos]).replace(' ', '').lower()
                if old_t != new_t:
                    results[i+shift+l] = pos
                    break
        shift += b - a - 1
    # setting the restulting tokens and their score values
    results_indices = topk.indices.gather(1, results)
    results_scores = topk.values.gather(1, results)
    # inset the chosen tokens
    inputs['input_ids'][masked_token_indices] = results_indices.flatten()

    text = tokenizer.decode(inputs['input_ids'].flatten())
    # set DataFrame
    df_indices = [[], [], [], []]
    df_data = {'score': []}
    # go over all masked tokens (full word tokenization)
    shift = 0# shifting value for the occurenc of subword tokens
    for i, m in enumerate(np.sort(mask_map, axis=0)):
        j, a, b = m
        for l in range(i+shift, i+shift+b-a):
            looked = results.flatten()[l].item()+1
            df_indices[0].extend(looked*[j])
            df_indices[1].extend(looked*[doc[j].text])
            df_indices[2].extend(looked*[doc[j].pos_])
            for r in range(looked):
                df_indices[3].append(tokenizer.decode(topk.indices[l,r].item()))
                df_data['score'].append(f'{topk.values[l,r].item():.4}')
        shift += b - a - 1
    # create the DataFrame
    df_MultiIndex = pd.MultiIndex.from_arrays(df_indices, names=['index', 'og_token', 'og_POS', 'token'])
    df = pd.DataFrame(df_data, index=df_MultiIndex)
    
    return text, df


def spin_text(text: str, tokenizer, model, mask_prob: float, max_prob: float=0.1, k: int=5, use_sub_tokens: bool=False, seed: Optional[int]=None) -> (str, pd.DataFrame):
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
        if text_decoded == work_text:
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
    if use_sub_tokens:
        for paragraph in paragraphs:
            token_ids = tokenizer(
                paragraph,
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors='pt'
            )
            # subword tokenization of the model tokenizer
            model_tok = [tokenizer.decode(t) for t in token_ids['input_ids'].flatten()]
            # fullword tokenization with POS tanging and name entity recognition
            doc = nlp(paragraph)
            # mapping betwen the two tokenizations
            mapping = tokenization_mapping(model_tok, doc)
            # "100% of words"
            len_tokens = len([1 for t in doc if check_token(t.text)])# TODO check if there is a better (more clear) option for 100% of tokens (e.g. len(doc))
            # how many tokens to mask
            N = min(max(int(len_tokens * mask_prob), 0), len(mapping))
            # maximum number of tokens that can be masked at once
            max_tokens = int(len_tokens * max_prob)
            # chose the tokens that should be replaced (so that at most max_tokens get replaced at once)
            mask_maps_idc = rng.choice(np.arange(len(mapping)), size=(max(N//max_tokens,1), min(max_tokens,N)), replace=False)
            mask_maps = mapping[mask_maps_idc]
            if N > max_tokens and N % max_tokens != 0:
                rest_map_idc = np.setdiff1d(np.arange(len(mapping)), mask_maps_idc)
                if N < len(mapping):
                    rest_map_idc = rng.choice(rest_map_idc, size=N%max_tokens, replace=False)
                rest_map = mapping[rest_map_idc]
                mask_maps = np.append(
                    mask_maps,
                    [np.append(rest_map, (max_tokens-N%max_tokens)*[[-1,-1,-1]]).reshape(max_tokens, -1)],
                    axis=0
                )
            # sort the mappings of the mask tokens
            mask_maps = [np.sort(mask_maps[i], axis=0) for i in range(len(mask_maps))]
            # tensor for storing the resulting tokens
            memorize_results = torch.empty(token_ids['input_ids'].shape, dtype=torch.int64).fill_(-100)
            # go over all mappings
            for mask_map in mask_maps:
                # remove placeholder ([-1,-1,-1]) from the mask map
                if mask_map[0,0] == -1:
                    mask_map = mask_map[mask_map != [-1,-1,-1]]
                    mask_map = mask_map.reshape(len(mask_map)//3, -1)
                # get the position of the tokens to mask
                masked_token_indices = torch.zeros(token_ids['input_ids'].shape, dtype=bool)
                for _, a, b in mask_map:
                    for i in range(a, b):
                        masked_token_indices[0,i] = True
                # overide all subword tokens coresponding to a samled token with the mask token 
                inputs = copy.deepcopy(token_ids)
                inputs['input_ids'][masked_token_indices] = tokenizer.mask_token_id
                # masking model
                preds = model(**inputs)
                # activate out put with softmax function
                activated_preds = torch.softmax(preds[0], dim=2)
                # filter out the predictions
                masked_preds = activated_preds[masked_token_indices]
                # get the top k predictions
                topk = torch.topk(masked_preds, k, dim=1)
                # choosing new tokens
                results = torch.zeros((topk.indices.shape[0], 1), dtype=int)
                shift = 0
                for i, m in enumerate(mask_map):
                    _, a, b = m
                    # for all predicted subword token chose the first that is different form the old token
                    for l in range(b - a):
                        old_t = model_tok[a+l].replace(' ', '').lower()
                        for pos in range(k):
                            new_t = tokenizer.decode(topk.indices[i+shift+l,pos]).replace(' ', '').lower()
                            if old_t != new_t:
                                results[i+shift+l] = pos
                                break
                    shift += b - a - 1
                # setting the restulting tokens and their score values
                results_indices = topk.indices.gather(1, results)
                # store the resluting token ids
                memorize_results[masked_token_indices] = results_indices.flatten()

                # create the DataFrame
                # the indices to the mask_map for all replaced full word repeated k * number of sub tokens times
                mask_map_df_idc = np.zeros(0, dtype=int)
                # top-k numbering
                postitions = np.zeros(0, dtype=object)
                for j,m in enumerate(mask_map):
                    mask_map_df_idc = np.append(mask_map_df_idc, np.tile([j], (m[2]-m[1])*k))
                    if m[2]-m[1] > 1:
                        # for words split into sub tokens
                        for i in range(1, m[2]-m[1]+1):
                            postitions = np.append(postitions, [f'{i}.{r}' for r in range(1, k+1)])
                    else:
                        postitions = np.append(postitions, [f'{r}' for r in range(1, k+1)])
                # add the mask indices
                df_np_array = np.array([mask_map[j,0]+df_index_shift for j in mask_map_df_idc]).reshape(1, -1)
                # add the original tokens
                df_np_array = np.append(df_np_array, [[doc[mask_map[j,0]].text for j in mask_map_df_idc]], axis=0)
                # add the original POS tags
                df_np_array = np.append(df_np_array, [[doc[mask_map[j,0]].pos_ for j in mask_map_df_idc]], axis=0)
                # add the position numbers of the suggested tokens
                df_np_array = np.append(df_np_array, postitions.reshape(1, -1), axis=0)
                # df_np_array = np.append(df_np_array, np.tile(np.arange(1, k+1), len(mask_map_df_idc)//k).reshape(1, -1), axis=0)
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
                    for i, m in enumerate(mask_map):
                        idx, a, b = m
                        t = doc[idx]
                        len_new_ts = 0
                        for j in range(b-a):
                            # the j-th sub token at position l coresponding to the i-th mask word
                            new_t = topk_str[i*k+l+j]
                            if j == 0 and idx > 0 and doc[idx-1].text_with_ws[-1] == ' ':
                                new_t = new_t.replace(' ', '')
                            start = t.idx+shift+len_new_ts
                            end = start+len(new_t)
                            len_new_ts += len(new_t)
                            topk_mask_positions.append((start, end))
                            topk_paragraph = topk_paragraph[:start] + new_t + topk_paragraph[start+len(t):]
                        shift += len_new_ts - len(t)
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
                            elif t.i == len(topk_doc)-1:
                                print('Achtung!!!')
                        new_pos[i*k+l] = topk_doc[topk_idx].pos_# POS for i-th token at the l-th position
                        i += 1
                df_np_array = np.append(df_np_array, new_pos.reshape(1, -1), axis=0)
                # add the scores
                df_np_array = np.append(df_np_array, topk.values.flatten().detach().numpy().reshape(1, -1), axis=0)
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
                    df = df.append(pd.DataFrame(df_np_array[4:].transpose() ,index=df_MultiIndex, columns=['token', 'POS', 'score', '']))
                else:
                    df = pd.DataFrame(df_np_array[4:].transpose() ,index=df_MultiIndex, columns=['token', 'POS', 'score', ''])

            # shift the indices for the DataFrame ot the next paragraph
            df_index_shift += len(doc)
            # inset the chosen tokens in the model tokenization
            new_token_indices = (memorize_results != -100)
            token_ids['input_ids'][new_token_indices] = memorize_results[new_token_indices]
            # decode the resluting token ids
            spun_paragraph = tokenizer.decode(token_ids['input_ids'].flatten())[3:-4]# slicing excludes the start and end tokens (adjust for other models)
            # store the spun paragraph
            spun_paragraphs.append(spun_paragraph)
    else:
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
            for mask_list in mask_idc:
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
                )
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
                results = torch.zeros((topk.indices.shape[0], 1), dtype=int)
                for i, m_idx in enumerate(mask_list):
                    old_t = doc[m_idx].text.lower()
                    for pos in range(k):
                        new_t = tokenizer.decode(topk.indices[i,pos]).replace(' ', '').lower()
                        if old_t != new_t:
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
                df_np_array = np.append(df_np_array, topk.values.flatten().detach().numpy().reshape(1, -1), axis=0)
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
                    df = df.append(pd.DataFrame(df_np_array[4:].transpose() ,index=df_MultiIndex, columns=['token', 'POS', 'score', '']))
                else:
                    df = pd.DataFrame(df_np_array[4:].transpose() ,index=df_MultiIndex, columns=['token', 'POS', 'score', ''])

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

    model_name = 'roberta-large'
    max_seq_len = 512
    mask_prob = 0.5
    k = 5
    seed = datetime.now().microsecond
    tokenizer, lm = init_model(model_name, max_seq_len)

    path = os.path.join(get_local_path(), *['data', 'wikipedia', 'ogUTF-8', '232530-ORIG-13.txt'])#232530-ORIG-13.txt, 1208667-ORIG-4.txt
    with open(path, 'r', encoding='utf-8') as file:
        toy_sentence = file.read()
    print(toy_sentence)

    # spun_text, df = spin_text_simple(toy_sentence, tokenizer, lm, mask_prob, k)
    # print(spun_text)
    # print(df)

    spun_text, df1 = spin_text(toy_sentence, tokenizer, lm, mask_prob, k=k, use_sub_tokens=True, seed=seed)

    print(spun_text)

    print(df1)

    spun_text, df2 = spin_text(toy_sentence, tokenizer, lm, mask_prob, k=k, use_sub_tokens=False, seed=seed)

    print(spun_text)

    print(df2)

    with open(os.path.join(get_local_path(), *['data', 'test.txt']), 'w', encoding='utf-8', newline='\n') as file:
        file.write(df1.sort_values(
            ['index', 'top-k'],
            key=lambda series : series.astype(int) if series.name == 'index' else series
        ).to_string())

        file.write('\n\n--------------------------\n\n')

        file.write(df2.sort_values(
            ['index', 'top-k'],
            key=lambda series : series.astype(int) if series.name == 'index' else series
        ).to_string())

