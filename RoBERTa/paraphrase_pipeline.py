import numpy as np
import pandas as pd
import torch as torch
from transformers import pipeline
from transformers.pipelines import FillMaskPipeline
from transformers import AutoTokenizer, AutoModelForMaskedLM
from transformers import BatchEncoding
from transformers.utils import logging
import nltk

import random as random
import copy as copy

logger = logging.get_logger(__name__)

# TODO: Add comments and ether block the use of tensorflow or make it also runable
class ParaphrasePipeline():
    
    def __init__(
        self,
        unmasker : FillMaskPipeline,
        input_window_size =  512
    ):
        self.tokenizer = unmasker.tokenizer
        self.model = unmasker.model
        self.device = unmasker.device
        # self.model.resize_token_embeddings(len(self.tokenizer))
        self.input_window_size = min(self.tokenizer.model_max_length, input_window_size)

    def unmasker_postproc(self, outputs, inputs, window_ids=None, targets=None, top_k=5):
        results = []
        batch_size = outputs.size(0)

        if targets is not None:
            if len(targets) == 0 or len(targets[0]) == 0:
                #raise ValueError("At least one target must be provided when passed.")
                print("At least one target must be provided when passed.")
                return None
            if isinstance(targets, str):
                targets = [targets]

            targets_proc = []
            for target in targets:
                target_enc = self.tokenizer.tokenize(target)
                if len(target_enc) > 1 or target_enc[0] == self.tokenizer.unk_token:
                    logger.warning(
                        "The specified target token `{}` does not exist in the model vocabulary. Replacing with `{}`.".format(
                            target, target_enc[0]
                        )
                    )
                targets_proc.append(target_enc[0])
            target_inds = np.array(self.tokenizer.convert_tokens_to_ids(targets_proc))

        for i in range(batch_size):
            input_ids = inputs["input_ids"][i]
            result = []

            masked_index_in = torch.nonzero(input_ids == self.tokenizer.mask_token_id, as_tuple=False)
            # check if the input was given to the model in a reduced form (with the window_ids)
            if window_ids == None:
                masked_index_out = masked_index_in
            else:
                masked_index_out = torch.nonzero(window_ids[i] == self.tokenizer.mask_token_id, as_tuple=False)

            # Fill mask pipeline supports only one ${mask_token} per sample
            # self.ensure_exactly_one_mask_token(masked_index.numpy())
            logits = outputs[i, masked_index_out.item(), :]
            probs = logits.softmax(dim=0)
            if targets is None:
                values, predictions = probs.topk(top_k)
            else:
                values = probs[..., target_inds]
                sort_inds = list(reversed(values.argsort(dim=-1)))
                values = values[..., sort_inds]
                predictions = target_inds[sort_inds]

            for v, p in zip(values.tolist(), predictions.tolist()):
                tokens = copy.deepcopy(input_ids.cpu().numpy())
                tokens[masked_index_in] = p
                # Filter padding out:
                tokens = tokens[np.where(tokens != self.tokenizer.pad_token_id)]
                result.append(
                    {
                        "sequence": self.tokenizer.decode(tokens),
                        "score": v,
                        "token": p,
                        "token_str": self.tokenizer.convert_ids_to_tokens(p),
                    }
                )
            
            # Append
            results += [result]

        if len(results) == 1:
            return results[0]
        return results

    def input_window(self, i, windowSize, textSize, input):
        shift = min(max(0, i+1 - windowSize // 2), textSize - windowSize)

        window_tensors = [input['input_ids'][:,0], input['input_ids'][0, 1+shift:windowSize-1+shift], input['input_ids'][:,-1]]
        window_input_ids = torch.cat(window_tensors).view(1, -1)
        window_atten_tensors = [input['attention_mask'][:,0], input['attention_mask'][0, 1+shift:windowSize-1+shift], input['attention_mask'][:,-1]]
        window_attention_mask = torch.cat(window_atten_tensors).view(1, -1)

        return BatchEncoding({'input_ids' : window_input_ids, 'attention_mask' : window_attention_mask})

    def ensure_tensor_on_device(self, **inputs):
        """
        Ensure PyTorch tensors are on the specified device.

        Args:
            inputs (keyword arguments that should be :obj:`torch.Tensor`): The tensors to place on :obj:`self.device`.

        Return:
            :obj:`Dict[str, torch.Tensor]`: The same as :obj:`inputs` but on the proper device.
        """
        return {name: tensor.to(self.device) for name, tensor in inputs.items()}

    def parapherase(self, og_text, mask=0.25, range_replace=(0, 5), use_score=False, replace_direct=False, mark_replace=False, return_df=False, startEndToken=False):
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
            replace_direct: determents wether the tokens should be replaced directly or not
            mark_replace: determents if in the returned spun text the new tokens are marked with square brackets
            return_df: determents if a DataFrame with information of the paraphrase process is returned (more info on the DataFrame see below)
            startEndToken: determents if the start and end tokens are removed for the spun text

        Return:
            :str: (when return_df=False) the paraphrased og_text
            or
            :str, DataFrame: (when return_df=True) the paraphrased og_text
                and a Pandas DataFrame witch contains for all replaced tokens
                the index in the text in order of there repleacement and
                for all by the unmasker returned candidats:
                    - token: the token ID
                    - token_str: the token string
                    - POS: the POS tag of the token
                    - score: the score given by the unmasker
                    - state: where the token was ignored, looked or chosen by the paraphraser
        """
        encode_input = self.tokenizer(og_text, return_tensors='pt')

        length = encode_input['input_ids'][0].size()[0]
        N = max(1, int(mask * length))# N tokens get replacest

        if type(range_replace) is tuple:
            range_replace = [range_replace]
        RRL = len(range_replace) # length of range_replace 

        replace_ids = random.sample([i for i in range(1, length - 1)], k=N)

        if return_df:
            # DataFrame
            indexArrays = [[], [], [], []]
            dfData = {'token_str' : [], 'POS' : [], 'score' : [], 'state' : []}
            # Tokenization for POS-tags
            tokenization = [self.tokenizer.decode(t) for t in encode_input['input_ids'][0,1:-1]]

        if replace_direct == False:
            replace = []
        for k, i in enumerate(replace_ids):
            if return_df:
                # Determent original POS of the to replace token
                og_pos = nltk.pos_tag(tokenization)[i-1][1]

            if replace_direct == False:
                tmp = copy.deepcopy(encode_input['input_ids'][0][i])
            encode_input['input_ids'][0][i] = self.tokenizer.mask_token_id
            
            if length > self.input_window_size:
                # input is to long for the model
                small_input = self.input_window(i, self.input_window_size, length, encode_input)
                with torch.no_grad():
                    small_input = self.ensure_tensor_on_device(**small_input)
                    output_tensor = self.model(**small_input)[0].cpu()
                output = self.unmasker_postproc(output_tensor, encode_input,
                                                window_ids=small_input['input_ids'].cpu(), top_k=range_replace[k%RRL][1])
            else:
                with torch.no_grad():
                    encode_input = self.ensure_tensor_on_device(**encode_input)
                    output_tensor = self.model(**encode_input)[0].cpu()
                output = self.unmasker_postproc(output_tensor, encode_input, top_k=range_replace[k%RRL][1])

            if use_score:
                scores = []
                for prop in output[range_replace[k%RRL][0]:]:
                    scores.append(prop['score'])
                chosen = random.choices([j for j in range(range_replace[k%RRL][0], len(output))], weights=scores, k=1)[0]
            else:
                chosen = random.sample([j for j in range(range_replace[k%RRL][0], len(output))], k=1)[0]
            newToken = output[chosen]
            
            if return_df:
                # add Data for DataFrame
                indexArrays[0].extend(range_replace[k%RRL][1]*[i])
                indexArrays[1].extend(range_replace[k%RRL][1]*[self.tokenizer.decode(tmp)])
                indexArrays[2].extend(range_replace[k%RRL][1]*[og_pos])
                indexArrays[3].extend([o['token'] for o in output])
                dfData['token_str'].extend([o['token_str'] for o in output])

                for o in output:# Determent POS-tags for the suggested tokens
                    tokenization[i-1] = o['token_str'].replace('Ġ', '')
                    dfData['POS'].append(nltk.pos_tag(tokenization)[i-1][1])

                # Correct the tokenization to the used encode_input
                if replace_direct:
                    tokenization[i-1] = newToken['token_str'].replace('Ġ', '')
                else:
                    tokenization[i-1] = self.tokenizer.decode(tmp).replace('Ġ', '')

                dfData['score'].extend([o['score'] for o in output])
                dfData['state'].extend(['chosen' if j == chosen else 'ignored' if j < range_replace[k%RRL][0]
                                        else 'looked' for j in range(range_replace[k%RRL][1])])

            if replace_direct:
                encode_input['input_ids'][0][i] = newToken['token']
            else:
                encode_input['input_ids'][0][i] = tmp
                replace.append(newToken['token'])

        if replace_direct == False:
            for j, i in enumerate(replace_ids):
                encode_input['input_ids'][0][i] = replace[j]

        tokens = encode_input['input_ids'][0].cpu().numpy()
        # Filter padding out:
        tokens = tokens[np.where(tokens != self.tokenizer.pad_token_id)]
        if mark_replace:
            replace_ids = np.sort(replace_ids)
            count = 0
            for i in replace_ids:
                tokens = np.insert(tokens, i+(2*count), values=self.tokenizer.encode('[')[1]) # insert '['
                tokens = np.insert(tokens, i+(2*count)+2, values=self.tokenizer.encode(']')[1]) # insert ']'
                count += 1
        newText = self.tokenizer.decode(tokens)

        startToken = self.tokenizer.bos_token
        endToken = self.tokenizer.eos_token
        if startEndToken:
            newText = newText.replace(startToken, '<s> ')
            newText = newText.replace(endToken, ' </s>')
        else:
            newText = newText.replace(startToken, '')
            newText = newText.replace(endToken, '')

        if return_df:
            # set DataFrame
            dfIndex = pd.MultiIndex.from_arrays(indexArrays, names=['index', 'og_token_str', 'og_POS', 'token'])
            df = pd.DataFrame(dfData, index=dfIndex)

            return newText, df
        else:
            return newText

# main for testing
if __name__ == "__main__":

    originalText = "The English Wikipedia was the first Wikipedia edition and has remained the largest. It has pioneered many ideas as conventions, policies or features which were later adopted by Wikipedia editions in some of the other languages."
    # originalText = "Hello I'm a good model."

    # filename = r"./Paraphraser/data/wikipedia/og/339-ORIG-2.txt"
    # filename = r"./Paraphraser/data/thesis/ogUTF-8/1-ORIG-18.txt"
    # with open(filename, 'r', encoding='utf-8') as file:
    #     originalText = file.read()

    tokenizer = AutoTokenizer.from_pretrained("roberta-large")
    model = AutoModelForMaskedLM.from_pretrained("roberta-large")
    model.resize_token_embeddings(len(tokenizer))
    unmasker = FillMaskPipeline(model=model, tokenizer=tokenizer, device=0)

    # unmasker = pipeline('fill-mask', model='roberta-large')
    paraphraser = ParaphrasePipeline(unmasker, input_window_size=512)

    spun_text, df = paraphraser.parapherase(originalText, mask=0.1, range_replace=[(1, 4), (0, 3)], mark_replace=True, return_df=True)

    print(spun_text)
    print(df)

    # spun_text = paraphraser.parapherase(originalText, mask=0.1, range_replace=(1, 4), mark_replace=True)

    # print(spun_text)
