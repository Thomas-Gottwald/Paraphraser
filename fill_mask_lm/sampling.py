import random
import os
from datetime import datetime
from tqdm import tqdm
from typing import Union

from .paraphraser import Data, Model, init_model, spin_text
from .getPath import get_local_path

def create_sample(sample_size: int, data: list, spin_text_args: list,
    model_type: Union[Model,str]=Model.ROBERTA, max_seq_len: int=512,
    disguise_sample: bool=False):
    """
    Creates a sample of spun texts

    Args:
        sample_size: The size of the sample
        data: List of dataset Enums or strings
        spin_text_args: list of arguments for the paraphraser
        model_type: Enum for the mask model for the paraphraser
            or a string referring to a mask language model that can
            be loaded by AutoModelForMaskLM from transformers
        max_seq_len: Maximum input size of the model
        disguise_sample: Wether the information if a text was spun or not is stored in the same file as the texts
            or in separate log file
    """
    # set up the language model
    tokenizer, lm = init_model(model_type, max_seq_len)
    # the name of the folder to stor this sample
    sampleFolder = datetime.now().strftime("%Y-%m-%d %H_%M_%S")
    # set the paths to the data
    path = get_local_path()
    data_paths = []
    for d in data:
        data_paths.append(os.path.join(path, *['data', str(d)]))
    # get the original text file names
    ogfiles = []
    for dp in data_paths:
        ogfiles.append([f for f in os.listdir(os.path.join(dp, 'ogUTF-8')) if os.path.isfile(os.path.join(dp, *['ogUTF-8', f]))])
    # chose the sample files
    sample_files = set()
    for i in range(sample_size):
        chose_data = random.randint(0, 10 * len(data)-1) // 10
        chose_file = random.sample(ogfiles[chose_data], k=1)[0]
        sample_files.add((data[chose_data], chose_file))
    # creating the folder for the sample if not exits
    if not os.path.exists(os.path.join(path, *['data', 'sample', sampleFolder])):
        os.makedirs(os.path.join(path, *['data', 'sample', sampleFolder]))

    if disguise_sample:
        # stores the information in witch order the text occur in the sample files (for disguise_sample=True)
        logInfo = []
    # write the samples
    for d, sf in tqdm(sample_files):
        # open the sample file
        sfile_name = sf.replace('ORIG', str(d).upper())
        sfile = open(os.path.join(path, *['data', 'sample', sampleFolder, sfile_name]), 'w', encoding='utf-8', newline='\n')
        # create DataFrame file
        dffile_name = 'df_' + sfile_name
        dffile = open(os.path.join(path, *['data', 'sample', sampleFolder, dffile_name]), 'w', encoding='utf-8', newline='\n')
        # the order in which the different spun_texts and the original text (index -1) are handeld
        order = [i for i in range(-1, len(spin_text_args))]

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
                    arg_info_text = ", ".join(['{}={}'.format(arg, spin_text_args[o][arg]) for arg in spin_text_args[o]])
                sfile.write(arg_info_text + ": \n\n")

            og_path = os.path.join(data_paths[data.index(d)], *['ogUTF-8', sf])
            with open(og_path, encoding='utf-8') as file:
                originalText = file.read()
            if o == -1:
                sfile.write(originalText)
            else:
                spun_text, df = spin_text(originalText, tokenizer, lm, **spin_text_args[o])
                sfile.write(spun_text)
                # Write in the DataFrame file
                arg_info_text = ", ".join(['{}={}'.format(arg, spin_text_args[o][arg]) for arg in spin_text_args[o]])
                dffile.write(arg_info_text + ": \n\n")
                dffile.write(df.sort_values(
                    ['index', 'top-k'],
                    key=lambda series : series.astype(int) if series.name == 'index' else series
                ).to_string())
                dffile.write('\n\n' + 100*'-' + '\n\n')

            sfile.write('\n\n' + 100*'-' + '\n\n')

        if disguise_sample:
            logInfo.append(sfileInfo)

        # close the sample file and DataFrame file
        sfile.close()
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
                        file.write("    " + ", ".join(['{}={}'.format(arg, spin_text_args[info[i]][arg]) for arg in spin_text_args[info[i]]]) + "\n")
                file.write('\n')
