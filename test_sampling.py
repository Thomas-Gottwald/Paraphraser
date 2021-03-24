from datetime import datetime

from fill_mask_lm.paraphraser import Data, Model
from fill_mask_lm.sampling import create_sample

# using one seed for all sample texts
seed = datetime.now().microsecond
create_sample(
    3,
    [Data.WIKIPEDIA],
    [{'mask_prob': 0.5, 'max_prob': 0.1, 'k': 5, 'seed': seed}],
    model_type=Model.ROBERTA,
    disguise_sample=False
)