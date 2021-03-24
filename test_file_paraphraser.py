from fill_mask_lm.paraphraser import Data, Model
from fill_mask_lm.file_paraphraser import paraphrase_dataset

# witch data will be paraphrased
data = Data.WIKIPEDIA
# how many files should be paraphrased
N = 10000
# the masked language model
model_type = Model.ROBERTA
max_seq_len = 512
# the parameters for the paraphraser
mask_prob = 0.5
max_prob = 0.1
k = 5
spin_text_args = {'mask_prob': mask_prob, 'max_prob': max_prob, 'k': k}

# spin the dataset
paraphrase_dataset(data, N, model_type, max_seq_len, spin_text_args)