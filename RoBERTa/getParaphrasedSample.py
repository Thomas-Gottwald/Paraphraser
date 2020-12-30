import paraphrase_pipeline as ppipe
from getPath import get_local_path
from transformers import pipeline
from transformers.pipelines import FillMaskPipeline
from transformers import AutoTokenizer, AutoModelForMaskedLM
import os
from enum import Enum
import random as random

class Data(Enum):
    THESIS = 1
    ARXIV = 2
    WIKIPEDIA = 3

def get_paraphraser(device=0, input_window_size=200):
    """
    returns the paraphraser object
    """
    # setting up the paraphraser
    tokenizer = AutoTokenizer.from_pretrained("roberta-large")
    model = AutoModelForMaskedLM.from_pretrained("roberta-large")
    model.resize_token_embeddings(len(tokenizer))
    unmasker = FillMaskPipeline(model=model, tokenizer=tokenizer, device=device)
    paraphraser = ppipe.ParaphrasePipeline(unmasker, input_window_size=input_window_size)

    return paraphraser

def spin_text(paraphraser, og_path, sp_path, mask_pc=0.1, range_replace=(1, 4)):
    """
    uses the given paraphraser to spin the text in og_path and write it to sp_path
    """
    with open(og_path, encoding='utf-8') as file:
        originalText = file.read()
    spun_text = paraphraser.parapherase(originalText, mask=mask_pc, range_replace=range_replace)
    with open(sp_path, 'w', encoding='utf-8', newline='\n') as file:
        file.write(spun_text)

# the size of the sample
N = 5
# from wich data should be sampled
data = [Data.THESIS]
# witch amounts (in %) of paraphrased words should the sample text contain
para_pc = {0, 10, 20, 30}

# make sure that the original files are in the sample
if 0 not in para_pc:
    para_pc.add(0)

# set the paths to the data
path = get_local_path()
data_paths = []
for d in data:
    if d == Data.THESIS:
        data_paths.append(path + "/data/thesis")
    elif  data == Data.ARXIV:
        data_paths.append(path + "/data/arxiv")
    elif data == Data.WIKIPEDIA:
        data_paths.append(path + "/data/wikipedia")
if len(data_paths) == 0:
    print('data is not specificied!')
    quit()

# get the original text file names
ogfiles = []
for dp in data_paths:
    ogfiles.append([f for f in os.listdir(dp + "/ogUTF-8") if os.path.isfile(os.path.join(dp + "/ogUTF-8", f))])

# chose the sample files
sample_files = set()
for i in range(N):
    chose_data = random.randint(0, 10 * len(data)-1) // 10
    chose_file = random.sample(ogfiles[chose_data], k=1)[0]
    sample_files.add((data[chose_data], chose_file))

# creating the folder for the sample if not exits
if not os.path.exists(path + "/data/sample"):
    os.makedirs(path + "/data/sample")

paraphraser = None
# write the samples
for d, sf in sample_files:
    # open the sample file
    sfile_name = sf.replace('ORIG', 'THESIS' if d == Data.THESIS else 'ARXIV' if d == Data.ARXIV else 'WIKIPEDIA')
    sfile = open(path + "/data/sample/" + sfile_name, 'w', encoding='utf-8', newline='\n')

    # shuffle the oder of the percentages
    shuffle_pc = list(para_pc)
    random.shuffle(shuffle_pc)
    for pc in shuffle_pc:
        if pc == 0:
            full_path = data_paths[data.index(d)] + "/ogUTF-8/" + sf
        else:
            spun_f = sf.replace('ORIG', 'SPUN')
            full_path = data_paths[data.index(d)] + "/sp({}%)/".format(pc) + spun_f

            # if the spun text dose not exist then create one
            if not os.path.isfile(os.path.join(full_path)):
                # sfile.write('is no file ' + full_path)
                if paraphraser == None:
                    paraphraser = get_paraphraser()
                # spin the original text
                og_path = data_paths[data.index(d)] + "/ogUTF-8/" + sf
                spin_text(paraphraser, og_path, full_path, pc/100)
        
        # write into the sample file
        with open(full_path, encoding='utf-8') as file:
            sfile.write(file.read())
        sfile.write('\n\n' + 100*'-' + '\n\n')

    # close the sample file
    sfile.close()
