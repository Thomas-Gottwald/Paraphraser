import paraphrase_pipeline as ppipe
from getPath import get_local_path
from transformers import pipeline
from transformers.pipelines import FillMaskPipeline
from transformers import AutoTokenizer, AutoModelForMaskedLM
import os
from enum import Enum
import random as random
from datetime import datetime

class Data(Enum):
    THESIS = 1
    ARXIV = 2
    WIKIPEDIA = 3

def get_paraphraser(device=0, input_window_size=512):
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

def spin_text(paraphraser, og_path, sp_path, mask_pc=0.1, add_args={'range_replace' : (1, 4)}):
    """
    uses the given paraphraser to spin the text in og_path and write it to sp_path or return it if sp_path == None
    """
    with open(og_path, encoding='utf-8') as file:
        originalText = file.read()
    spun_text = paraphraser.parapherase(originalText, mask=mask_pc, **add_args)
    if sp_path != None:
        with open(sp_path, 'w', encoding='utf-8', newline='\n') as file:
            file.write(spun_text)
    else:
        return spun_text

# determents whether the paraphrased text are loaded from the data if they exist or created new
load_spun_form_data = False
# additional arguments for the paraphraser (only used with load_spun_form_data = False)
paraphrase_args = {'range_replace' : [(1, 4), (0, 4)], 'use_score': False, 'mark_replace': False, 'return_df': False, 'startEndToken': False}
# the size of the sample
N = 20
# from witch data should be sampled
data = [Data.WIKIPEDIA]
# witch amounts (in %) of paraphrased words should the sample text contain
para_pc = {0, 10, 20, 30}
# the name of the folder to stor this sample
sampleFolder = datetime.now().strftime("%Y-%m-%d %H_%M_%S")

# make sure that the original files are in the sample
if 0 not in para_pc:
    para_pc.add(0)

# set the paths to the data
path = get_local_path()
data_paths = []
for d in data:
    if d == Data.THESIS:
        data_paths.append(os.path.join(path, *['data', 'thesis']))
    elif  d == Data.ARXIV:
        data_paths.append(os.path.join(path, *['data', 'arxiv']))
    elif d == Data.WIKIPEDIA:
        data_paths.append(os.path.join(path, *['data', 'wikipedia']))
if len(data_paths) == 0:
    print('data is not specificied!')
    quit()

# get the original text file names
ogfiles = []
for dp in data_paths:
    ogfiles.append([f for f in os.listdir(os.path.join(dp, 'ogUTF-8')) if os.path.isfile(os.path.join(dp, *['ogUTF-8', f]))])

# chose the sample files
sample_files = set()
for i in range(N):
    chose_data = random.randint(0, 10 * len(data)-1) // 10
    chose_file = random.sample(ogfiles[chose_data], k=1)[0]
    sample_files.add((data[chose_data], chose_file))

# creating the folder for the sample if not exits
if not os.path.exists(os.path.join(path, *['data', 'sample', sampleFolder])):
    os.makedirs(os.path.join(path, *['data', 'sample', sampleFolder]))

# check if all the spun text directories exist (if not create them so that the spun files can be created)
if load_spun_form_data:
    for dp in data_paths:
        for pc in para_pc:
            if pc != 0 and not os.path.exists(os.path.join(dp, 'sp({}%)'.format(pc))):
                os.makedirs(os.path.join(dp, 'sp({}%)'.format(pc)))

# stores the information in witch order the text occur in the sample files 
logInfo = []

paraphraser = None
if not load_spun_form_data:
    paraphraser = get_paraphraser()
# write the samples
for d, sf in sample_files:
    # open the sample file
    sfile_name = sf.replace('ORIG', 'THESIS' if d == Data.THESIS else 'ARXIV' if d == Data.ARXIV else 'WIKIPEDIA')
    sfile = open(os.path.join(path, *['data', 'sample', sampleFolder, sfile_name]), 'w', encoding='utf-8', newline='\n')

    # information in witch order the text occur this sample file
    sfileInfo = [sfile_name]

    # shuffle the oder of the percentages
    shuffle_pc = list(para_pc)
    random.shuffle(shuffle_pc)
    for pc in shuffle_pc:
        sfileInfo.append(pc)# remember the percentage of replaced words
        if pc == 0:
            full_path = os.path.join(data_paths[data.index(d)], *['ogUTF-8', sf])
        elif load_spun_form_data:
            spun_f = sf.replace('ORIG', 'SPUN')
            full_path = os.path.join(data_paths[data.index(d)], *['sp({}%)'.format(pc), spun_f])

            # if the spun text dose not exist then create one
            if not os.path.isfile(full_path):
                if paraphraser == None:
                    paraphraser = get_paraphraser()
                # spin the original text
                og_path = os.path.join(data_paths[data.index(d)], *['ogUTF-8', sf])
                spin_text(paraphraser, og_path, full_path, pc/100)
        
        # write into the sample file
        if load_spun_form_data or pc == 0:
            with open(full_path, encoding='utf-8') as file:
                sfile.write(file.read())
        else:
            # spin the original text
            og_path = os.path.join(data_paths[data.index(d)], *['ogUTF-8', sf])
            sfile.write(spin_text(paraphraser, og_path, None, pc/100))
        sfile.write('\n\n' + 100*'-' + '\n\n')

    logInfo.append(sfileInfo)

    # close the sample file
    sfile.close()

# create the Info file where the information for the sample is stored
with open(os.path.join(path, *['data', 'sample', sampleFolder, 'Info.txt']), 'w', encoding='utf-8', newline='\n') as file:
    if not load_spun_form_data:
        file.write(', '.join(['{}={}'.format(arg, paraphrase_args[arg]) for arg in paraphrase_args]))
        file.write('\n\n')
    for info in logInfo:
        file.write(info[0] + '\n')
        for i in range(1, len(info)):
            file.write("    {}%\n".format(info[i]))
        file.write('\n')