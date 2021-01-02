import paraphrase_pipeline as ppipe
from getPath import get_local_path
from transformers import pipeline
from transformers.pipelines import FillMaskPipeline
from transformers import AutoTokenizer, AutoModelForMaskedLM
import os
from enum import Enum
from tqdm import tqdm

class Data(Enum):
    THESIS = 1
    ARXIV = 2
    WIKIPEDIA = 3

# witch data will be paraphrased
data = Data.THESIS
# how much (in %) of the text will be replaced
mask_pc = 20 # 10%, 20%, 30%

# setting up the paraphraser
tokenizer = AutoTokenizer.from_pretrained("roberta-large")
model = AutoModelForMaskedLM.from_pretrained("roberta-large")
model.resize_token_embeddings(len(tokenizer))
unmasker = FillMaskPipeline(model=model, tokenizer=tokenizer, device=0)
# unmasker = pipeline('fill-mask', model='roberta-large')
paraphraser = ppipe.ParaphrasePipeline(unmasker, input_window_size=200)

# setting the path to the data
path = get_local_path()
if data == Data.THESIS:
    ogpath = path + "/data/thesis/ogUTF-8"
    sppath = path + "/data/thesis"
elif  data == Data.ARXIV:
    ogpath = path + "/data/arxiv/ogUTF-8"
    sppath = path + "/data/arxiv"
elif data == Data.WIKIPEDIA:
    ogpath = path + "/data/wikipedia/ogUTF-8"
    sppath = path + "/data/wikipedia"
else:
    print('data is not specificied!')
    quit()

# getting the names of the original text files
ogfiles = [f for f in os.listdir(ogpath) if os.path.isfile(os.path.join(ogpath, f))]

# creating the folder for the spun files if it not exist
sp_dir = "sp({}%)".format(mask_pc)
sppath = os.path.join(sppath, sp_dir)
if not os.path.exists(sppath):
    os.makedirs(sppath)

N = len(ogfiles)
for i in tqdm(range(N)):
    with open(os.path.join(ogpath, ogfiles[i]), encoding='utf-8') as file:
        originalText = file.read()
    spun_text = paraphraser.parapherase(originalText, mask=mask_pc/100, range_replace=(1, 4))
    spunfile = ogfiles[i].replace('ORIG', 'SPUN')
    with open(os.path.join(sppath, spunfile), 'w', encoding='utf-8', newline='\n') as file:
        file.write(spun_text)
