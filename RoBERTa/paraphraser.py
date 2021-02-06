import torch
import spacy
import re
import numpy as np
import pandas as pd
import random
from transformers import AutoConfig, AutoTokenizer, AutoModelForMaskedLM

from getPath import get_local_path
import os

def init_model(model_name_or_path: str, max_len: int):
    config = AutoConfig.from_pretrained(model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, model_max_length=max_len)
    masked_model = AutoModelForMaskedLM.from_pretrained(model_name_or_path, config=config)

    return tokenizer, masked_model

def check_token(token: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z]+([-'][A-Za-z]+)*", token))

def tokenization_mapping(model_tok : list, doc) -> list:
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
                return mapping

        # if t is next part of doc[j].text
        if doc[j].text.startswith(t, len(token), len(token)+len(t)):
            # if t is the first part of doc[j].text (then len(token)=0 => token='')
            if token == '':
                start = i
                token = t
            else:
                token += t
        else:
            # if not set token againg to the empty string
            token = ''

        # if all parts of doc[j].text were found
        if token == doc[j].text:
            mapping.append([j, start, i+1])
            j += 1
            if j >= len(doc):
                break
            token = ''

    return mapping

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


def spin_text(paragraph: str, tokenizer, model, mask_prob: float, k: int=5) -> (str, pd.DataFrame):
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
    df = pd.DataFrame(df_data, index= df_MultiIndex)
    
    return text, df

if __name__ == '__main__':
    model_name = 'roberta-large'
    max_seq_len = 512
    mask_prob = 0.5
    k = 5
    tokenizer, lm = init_model(model_name, max_seq_len)

    path = os.path.join(get_local_path(), *['data', 'wikipedia', 'ogUTF-8', '1208667-ORIG-4.txt'])
    with open(path, 'r', encoding='utf-8') as file:
        toy_sentence = file.read()
    print(toy_sentence)

    spun_text, df = spin_text(toy_sentence, tokenizer, lm, mask_prob, k)
    print(spun_text)
    print(df)
    