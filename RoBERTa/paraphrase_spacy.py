import numpy as np
import pandas as pd
import torch as torch
from transformers import pipeline
from transformers.pipelines import FillMaskPipeline
from transformers import AutoTokenizer, AutoModelForMaskedLM
import spacy as spacy

import random as random
import copy as copy

import time


class ParaphraseSpacy():

    def __init__(
        self,
        unmasker : FillMaskPipeline,
    ):
        self.tokenizer = unmasker.tokenizer
        self.unmasker = unmasker
        self.max_length = self.tokenizer.model_max_length
        self.split_size = (9 * self.max_length) // 10

    def __call__(self, og_text, mask=0.25, range_replace=(0, 5), use_score=False, return_df=False):
        """
        paraphases the og_text

        Args:
            og_text: the original text to paraphrase
            mask: amount of words that should be replaced by the paraphraser between 0 and 1 (at least on word will be replaced)
            range_replace: a tuple of natural numbers (n, m) or a list of such tuples
                The tuples determent that for any token that is replaced m tokens will be given by the unmasker and
                the fist n will be ignord for the choice of the new token.
                If more tupels in a list are past then each tuple will be used equally often (by appling one after another and loping back
                when the end of the list is reached)
            use_score: determents if for the choice of the the new tokens the scores given by the unmasker are used as weights
            return_df: determents if a DataFrame with information of the paraphrase process is returned (more info on the DataFrame see below)

        Return:
            :str: (when return_df=False) the paraphrased og_text
            or
            :str, DataFrame: (when return_df=True) the paraphrased og_text
                and a Pandas DataFrame witch contains for all replaced tokens
                the index in the text in order of there repleacement and
                for all by the unmasker returned candidats:
                    - token_str: the token string
                    - POS: the POS tag of the token
                    - score: the score given by the unmasker
                    - state: where the token was ignored, looked or chosen by the paraphraser
        """
        token_len = len(self.tokenizer(og_text)['input_ids'])
        if token_len <= self.max_length:
            return self._parapherase(og_text, mask, range_replace, use_score, return_df)
        else:
            nlp = spacy.load("en_core_web_sm", disable=["parser", "tagger", "ner"])
            nlp.add_pipe(spacy.lang.en.English().create_pipe("sentencizer"))

            doc = nlp(og_text)
            sents = list(doc.sents)

            og_texts = []
            store_spaces = []

            divide = token_len // self.split_size + 1

            pos = 0

            while len(og_texts) < divide:
                if divide > len(sents):
                    print("At least one to long sentence!!!")
                    return None
                rest = len(sents) % divide
                shift = 0
                for j in range(divide):
                    start = sents[j*(len(sents)//divide)+shift].start
                    if rest > 0:
                        rest -= 1
                        shift += 1
                    stop = sents[(j+1)*(len(sents)//divide)-1+shift].end
                    tmp_text = doc[start:stop].text
                    
                    if len(tokenizer(tmp_text)['input_ids']) > self.max_length:
                        divide += 1
                        og_texts = []
                        store_spaces = []
                        break
                    pos += len(tmp_text)
                    if pos < len(og_text):
                        if og_text[pos] == " ":
                            store_spaces.append(" ")
                            pos += 1
                        else:
                            store_spaces.append("")
                    og_texts.append(tmp_text)

            spun_texts = []
            for t in og_texts:
                spun_texts.append(self._parapherase(t, mask, range_replace, use_score, return_df))
            
            if return_df:
                spun_text = spun_texts[0][0]
                for j in range(len(store_spaces)):
                    spun_text += store_spaces[j] + spun_texts[j+1][0]

                df = pd.concat([st[1] for st in spun_texts])

                return spun_text, df
            else:
                spun_text = spun_texts[0]
                for j in range(len(store_spaces)):
                    spun_text += store_spaces[j] + spun_texts[j+1]

                return spun_text

    def _parapherase(self, og_text, mask=0.25, range_replace=(0, 5), use_score=False, return_df=False):
        text = og_text

        if type(range_replace) is tuple:
            range_replace = [range_replace]
        RRL = len(range_replace) # length of range_replace

        nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])# tokenizer with POS tagging
        og_doc = nlp(text)# tokenize original text

        length = len(og_doc)# length of text tokenization
        N = max(1, int(mask * length))# N tokens get replacest

        mask = self.tokenizer.mask_token

        replace_indices = np.sort(random.sample([i for i in range(1, length - 1)], k=N))

        if return_df:# Objects for DataFrame data
            indexArrays = [[], [], [], []]
            dfData = {'POS' : [], 'score' : [], 'state' : []}

        replace = []
        for i, index in enumerate(replace_indices):
            n = 0 if index == 0 else len(og_doc[:index].text)
            m = len(og_doc[index])
            lsp = 0
            if text[n] == " ":
                lsp = 1
            tmp_text = text[:n+lsp] + mask + text[n+lsp+m:]

            output = self.unmasker(tmp_text, top_k=range_replace[i%RRL][1])

            if use_score:
                scores = []
                for prop in output[range_replace[i%RRL][0]:]:
                    scores.append(prop['score'])
                chosen = random.choices([j for j in range(range_replace[i%RRL][0], len(output))], weights=scores, k=1)[0]
            else:
                chosen = random.sample([j for j in range(range_replace[i%RRL][0], len(output))], k=1)[0]
            
            for o in output:
                o['token_str'] = o['token_str'].replace("Ġ", " ")# the model tokenizer returns leading spaces as "Ġ" ["Ġ"=b'\xc4\xa0'.decode('utf-8')]
            new_token_str = output[chosen]['token_str']

            if return_df:# add data to DataFrame
                indexArrays[0].extend(range_replace[i%RRL][1]*[index])
                indexArrays[1].extend(range_replace[i%RRL][1]*[og_doc[index].text])
                indexArrays[2].extend(range_replace[i%RRL][1]*[og_doc[index].pos_])
                indexArrays[3].extend([o['token_str'] for o in output])

                # determen POS for all surgested tokens
                for o in output:
                    tmp_text = text[:n+(lsp if o['token_str'][0] != " " else 0)] + o['token_str'] + text[n+lsp+m:]
                    tmp_doc = nlp(tmp_text)
                    dfData['POS'].append(tmp_doc[index].pos_)

                dfData['score'].extend([o['score'] for o in output])
                dfData['state'].extend(['chosen' if j == chosen else 'ignored' if j < range_replace[i%RRL][0]
                                        else 'looked' for j in range(range_replace[i%RRL][1])])

            replace.append([new_token_str, n, n+m+lsp, (lsp if new_token_str[0] != " " else 0)])

        new_text = text
        for i in np.argsort(-1*replace_indices):
            new_text = new_text[:replace[i][1]+replace[i][3]] + replace[i][0] + new_text[replace[i][2]:]

        if return_df:
            # create DataFrame
            dfIndices = pd.MultiIndex.from_arrays(indexArrays, names=['index', 'og_token', 'og_POS', 'token'])
            df = pd.DataFrame(dfData, dfIndices)

            return new_text, df
        else:
            return new_text


if __name__ == "__main__":

    tokenizer = AutoTokenizer.from_pretrained("roberta-large")
    model = AutoModelForMaskedLM.from_pretrained("roberta-large")
    model.resize_token_embeddings(len(tokenizer))
    unmasker = FillMaskPipeline(model=model, tokenizer=tokenizer, device=0)

    paraphraser = ParaphraseSpacy(unmasker)


    # originalText = "The English Wikipedia was the first Wikipedia edition and has remained the largest. It has pioneered many ideas as conventions, policies or features which were later adopted by Wikipedia editions in some of the other languages."
    filename = r"./Paraphraser/data/wikipedia/ogUTF-8/4115833-ORIG-15.txt"
    with open(filename, 'r', encoding='utf-8') as file:
        originalText = file.read()

    ###############
    # filename = r"./Paraphraser/data/wikipedia/ogUTF-8/232530-ORIG-13.txt"
    # with open(filename, 'r', encoding='utf-8') as file:
    #     originalText = file.read()

    # 1208667-ORIG-4.txt (92 tokens)
    # time: 47.73720383644104s
    # df: 52.59357261657715s
    # 232530-ORIG-13.txt (546 tokens)
    # time: 361.0956211090088s 391.4179744720459s
    # df: 464.22740292549133s
    # 4115833-ORIG-15.txt (635 tokens)

    # 39871658-ORIG-10.txt (1939 tokens)
  
    # start = time.time()
    # for _ in range(100):
    #     spun_text = paraphraser(originalText, mask=0.1, range_replace=[(1, 4), (0, 3)])
    # stop = time.time()
    # print("time: {}s".format(stop - start))

    # start = time.time()
    # for _ in range(100):
    #     spun_text, df = paraphraser(originalText, mask=0.1, range_replace=[(1, 4), (0, 3)], return_df=True)
    # stop = time.time()
    # print("df: {}s".format(stop - start))
    # quit()
    ##############

    # spunText = paraphraser(originalText, mask=0.1, range_replace=[(1, 4), (0, 4)])
    # print(spunText)

    spunText, df = paraphraser(originalText, mask=0.1, range_replace=[(1, 4), (0, 4)], return_df=True)
    print(spunText)
    print(df)
