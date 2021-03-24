import os
from datetime import datetime

from fill_mask_lm.paraphraser import init_model, spin_text, Model, Data
from fill_mask_lm.getPath import get_local_path

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