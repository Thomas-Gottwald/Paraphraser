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
data = Data.WIKIPEDIA
# how much (in %) of the text will be replaced
mask_pc = 30 # 10%, 20%, 30%

# the parameters for the Paraphraser
paraphrase_args = {'mask' : mask_pc/100.0 , 'range_replace' : [(1, 4), (0, 4)]}

# setting up the paraphraser
tokenizer = AutoTokenizer.from_pretrained("roberta-large")
model = AutoModelForMaskedLM.from_pretrained("roberta-large")
model.resize_token_embeddings(len(tokenizer))
unmasker = FillMaskPipeline(model=model, tokenizer=tokenizer, device=0)
# unmasker = pipeline('fill-mask', model='roberta-large')
paraphraser = ppipe.ParaphrasePipeline(unmasker, input_window_size=512)

# setting the path to the data
path = get_local_path()
if data == Data.THESIS:
    ogpath = os.path.join(path, *['data', 'thesis', 'ogUTF-8'])
    sppath = os.path.join(path, *['data', 'thesis'])
elif  data == Data.ARXIV:
    ogpath = os.path.join(path, *['data', 'arxiv', 'ogUTF-8'])
    sppath = os.path.join(path, *['data', 'arxiv'])
elif data == Data.WIKIPEDIA:
    ogpath = os.path.join(path, *['data', 'wikipedia', 'ogUTF-8'])
    sppath = os.path.join(path, *['data', 'wikipedia'])
else:
    print('data is not specificied!')
    quit()

# getting the names of the original text files
ogfiles = [f for f in os.listdir(ogpath) if os.path.isfile(os.path.join(ogpath, f))]

# creating the folder for the spun files if it not exist
# sp_dir = "sp({}%)".format(mask_pc) TODO
# generating the folder name for the spun files
sp_dir = "sp("
for key in paraphrase_args:
    if key == 'mask':
        sp_dir += "{}%, ".format(int(paraphrase_args.get(key) * 100))
    elif key == 'range_replace':
        sp_dir += "{}, ".format(paraphrase_args.get(key))
    elif paraphrase_args.get(key) == True:
        sp_dir += "{}, ".format(key)
    else:
        sp_dir += "{}: {}, ".format(key, paraphrase_args.get(key))
sp_dir = sp_dir[:-2] + ")"

sppath = os.path.join(sppath, sp_dir)
if not os.path.exists(sppath):
    os.makedirs(sppath)

phase = 2
phase_size = 5000
start = phase_size * (phase - 1)
stop = min(phase_size * phase, len(ogfiles))
# N = len(ogfiles) TODO
# for i in tqdm(range(N)): TODO
for i in tqdm(range(start, stop)):
    with open(os.path.join(ogpath, ogfiles[i]), encoding='utf-8') as file:
        originalText = file.read()
    # spun_text = paraphraser.parapherase(originalText, mask=mask_pc/100, range_replace=[(1, 4), (0, 4)]) TODO
    spun_text = paraphraser.parapherase(originalText, **paraphrase_args)
    spunfile = ogfiles[i].replace('ORIG', 'SPUN')
    with open(os.path.join(sppath, spunfile), 'w', encoding='utf-8', newline='\n') as file:
        file.write(spun_text)

# Wikipedia: 10%, [(1,4),(0,4)]
# phase 1: 10000/10000 [1:41:05<00:00,  1.65it/s]
# phase 2: 10000/10000 [1:47:49<00:00,  1.55it/s]
# phase 3: 10000/10000 [1:58:28<00:00,  1.41it/s]
# phase 4: 9241/9241 [1:37:37<00:00,  1.58it/s]

# Wikipedia: 20%, [(1,4),(0,4)]
# phase 1: 10000/10000 [3:27:10<00:00,  1.24s/it]
# phase 2: 10000/10000 [3:52:35<00:00,  1.40s/it]
# phase 3: 10000/10000 [3:25:21<00:00,  1.23s/it]
# phase 4: 9241/9241 [3:14:20<00:00,  1.26s/it]

# Wikipedia: 30%, [(1,4),(0,4)]
# phase 1: 5000/5000 [2:38:08<00:00,  1.90s/it]
# phase 2: 5000/5000 [2:35:40<00:00,  1.87s/it]