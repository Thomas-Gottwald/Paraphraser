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

def spin_text(paraphraser, og_path, para_args={'range_replace' : (1, 4)}):
    """
    uses the given paraphraser to spin the text in og_path and return the spun_text
    """
    with open(og_path, encoding='utf-8') as file:
        originalText = file.read()
    spun_text = paraphraser.parapherase(originalText, **para_args)
 
    return spun_text

# the size of the sample
N = 3
# from witch data should be sampled
data = [Data.WIKIPEDIA]
# arguments for the paraphraser
# (return_df=True is only aviable for load_spun_form_data=False and if it is True for all args)
paraphrase_args = [{'mask' : 0.1, 'range_replace' : [(1, 4), (0, 4)], 'return_df': True},
                   {'mask' : 0.2, 'range_replace' : [(1, 4), (0, 4)], 'return_df': True},
                   {'mask' : 0.3, 'range_replace' : [(1, 4), (0, 4)], 'return_df': True}]
# determents whether the paraphrased text are loaded from the data created new
load_spun_form_data = False
# determents whether it is hidden if a text was paraphrased
disguise_sample = False
# the name of the folder to stor this sample
sampleFolder = datetime.now().strftime("%Y-%m-%d %H_%M_%S")

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
    print('Data is not specificied!')
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

# check if return_df is ether True or False (dose not exist) for all paraphrase_args
return_df = False
count_df = 0
for arg in paraphrase_args:
    if 'return_df' in list(arg.keys()) and arg['return_df']:
        count_df += 1
if count_df == len(paraphrase_args) and not load_spun_form_data:
    return_df = True
elif count_df != 0 or load_spun_form_data:
    print('Return_df=True can only be used with load_from_data=False and ruturn_df=True for all args!')
    quit()

if disguise_sample:
    # stores the information in witch order the text occur in the sample files (for disguise_sample=True)
    logInfo = []

# paraphraser = None
# if not load_spun_from data set up the paraphraser
if not load_spun_form_data:
    paraphraser = get_paraphraser()
# write the samples
for d, sf in sample_files:
    # open the sample file
    sfile_name = sf.replace('ORIG', 'THESIS' if d == Data.THESIS else 'ARXIV' if d == Data.ARXIV else 'WIKIPEDIA')
    sfile = open(os.path.join(path, *['data', 'sample', sampleFolder, sfile_name]), 'w', encoding='utf-8', newline='\n')

    if return_df:
        # create DataFrame file
        dffile_name = 'df_' + sf.replace('ORIG', 'THESIS' if d == Data.THESIS else 'ARXIV' if d == Data.ARXIV else 'WIKIPEDIA')
        dffile = open(os.path.join(path, *['data', 'sample', sampleFolder, dffile_name]), 'w', encoding='utf-8', newline='\n')

    # the order in which the different spun_texts and the original text (index -1) are handeld
    order = [i for i in range(-1, len(paraphrase_args))]

    if disguise_sample:
        # information in witch order the text occur this sample file
        sfileInfo = [sfile_name]

        # shuffle the oder (with -1 for the original text)
        random.shuffle(order)

    for o in order:
        if disguise_sample:
            sfileInfo.append(o)# remember the order
        else:
            if o == -1:
                arg_info_text = "Original Text"
            else:
                arg_info_text = ", ".join(['{}={}'.format(arg, paraphrase_args[o][arg]) for arg in paraphrase_args[o]])
            sfile.write(arg_info_text + ": \n\n")

        if o == -1 or not load_spun_form_data:
            full_path = os.path.join(data_paths[data.index(d)], *['ogUTF-8', sf])
        else:
            spun_f =  sf.replace('ORIG', 'SPUN')
            # generating the folder name for the spun files
            spun_dir = "sp("
            for key in paraphrase_args[o]:
                if key == 'mask':
                    spun_dir += "{}%, ".format(int(paraphrase_args[o].get(key) * 100))
                elif key == 'range_replace':
                    spun_dir += "{}, ".format(paraphrase_args[o].get(key))
                elif paraphrase_args[o].get(key) == True:
                    spun_dir += "{}, ".format(key)
                else:
                    spun_dir += "{}: {}, ".format(key, paraphrase_args[o].get(key))
            spun_dir = spun_dir[:-2] + ")"

            full_path = os.path.join(data_paths[data.index(d)], *[spun_dir, spun_f])

        if load_spun_form_data:
            with open(full_path, encoding='utf-8') as file:
                sfile.write(file.read())
        else:
            if o == -1:
                with open(full_path, encoding='utf-8') as file:
                    sfile.write(file.read())
            else:
                spun_text = spin_text(paraphraser, full_path, paraphrase_args[o])
                if return_df:
                    spun_text, df = spun_text# split the paraphraser out put in spun_text and DataFrame
                    
                    # Write in the DataFrame file
                    arg_info_text = ", ".join(['{}={}'.format(arg, paraphrase_args[o][arg]) for arg in paraphrase_args[o]])
                    dffile.write(arg_info_text + ": \n\n")
                    dffile.write(df.to_string())
                    dffile.write('\n\n' + 100*'-' + '\n\n')

                sfile.write(spun_text)
            
        sfile.write('\n\n' + 100*'-' + '\n\n')

    if disguise_sample:
        logInfo.append(sfileInfo)

    # close the sample file
    sfile.close()

    if return_df:
        # close DataFrame file
        dffile.close()

if disguise_sample:
    # create the Info file where the information for the sample is stored
    with open(os.path.join(path, *['data', 'sample', sampleFolder, 'Info.txt']), 'w', encoding='utf-8', newline='\n') as file:
        for info in logInfo:
            file.write(info[0] + '\n')
            for i in range(1, len(info)):
                if info[i] == -1:
                    file.write("    original Text\n")
                else:
                    file.write("    " + ", ".join(['{}={}'.format(arg, paraphrase_args[info[i]][arg]) for arg in paraphrase_args[info[i]]]) + "\n")
            file.write('\n')
