import numpy as np
import pandas as pd
import os
from typing import Optional, Union
from tqdm import tqdm

from .paraphraser import Data, Model, init_model, spin_text
from .getPath import get_local_path

def add_to_df_dataset(df: pd.DataFrame, df_dataset: Optional[pd.DataFrame]) -> (pd.DataFrame):
    """
    extracts information out of the DataFrame df consisting of the number of chosen and suggested
    tokens with certain POS tags and also the sum of the chosen and suggested token scores
    each grouped by the original POS tags
    this information is returned if df_dataset is None
    otherwise it is summed up with df_dataset and then returned
    """
    if type(df_dataset) is pd.DataFrame:
        df_suggested = df.reset_index().astype({'score': 'float64'}).groupby(['og_POS', 'POS']).agg(
            {'token': 'count', 'score': 'sum'}
        ).rename(columns={'token': 'suggested', 'score': 'sum_sug_score'})

        df_chosen = df.query('chosen == "x"').reset_index().astype({'score': 'float64'}).groupby(['og_POS', 'POS']).agg(
            {'chosen': 'count', 'score': 'sum'}
        ).rename(columns={'score': 'sum_cho_score'})

        df_dataset = df_dataset.add(df_suggested, fill_value=0)
        df_dataset = df_dataset.add(df_chosen, fill_value=0)
    else:
        df_dataset = df.reset_index().astype({'score': 'float64'}).groupby(['og_POS', 'POS']).agg(
            {'token': 'count', 'score': 'sum'}
        ).rename(columns={'token': 'suggested', 'score': 'sum_sug_score'})

        df_chosen = df.query('chosen == "x"').reset_index().astype({'score': 'float64'}).groupby(['og_POS', 'POS']).agg(
            {'chosen': 'count', 'score': 'sum'}
        ).rename(columns={'score': 'sum_cho_score'})

        df_dataset[['chosen', 'sum_cho_score']] = df_chosen[['chosen', 'sum_cho_score']]

    return df_dataset

def paraphrase_dataset(data: Union[Data,str], N: int, model_type: Union[Model,str], max_seq_len: int, spin_text_args: dict):
    """
    Paraphrases the dataset referred with data

    Args:
        data: Enum for the datasets (wikipedia, arxiv, thesis)
            or a string reffering to a dataset
        N: The number of files to paraphrase
        model_type: Enum for the mask neural language model
            or a string referring to a mask language model that can
            be loaded by AutoModelForMaskLM from transformers
        max_seq_len: The maximum length for input sequences for the model
        spin_text_args: the arguments for the function spin_text besides
            the text to spin, the model tokenizer and the language model
    """
    # load the language model and its tokenizer
    tokenizer, lm = init_model(model_type, max_seq_len)

    # setting the path to the data
    path = get_local_path()
    og_path = os.path.join(path, *['data', str(data), 'ogUTF-8'])
    sp_path = os.path.join(path, *['data', str(data)])

    # the folder name for the spun files
    sp_dir = "sp({},{})".format(str(model_type).replace('/', '_'), spin_text_args['mask_prob'])
    # create the folder if it dose not exist already
    sp_path = os.path.join(sp_path, sp_dir)
    if not os.path.exists(sp_path):
        os.makedirs(sp_path)
        os.makedirs(os.path.join(sp_path, 'text'))
        os.makedirs(os.path.join(sp_path, 'df'))
    
    # check if the parameter text files already exists
    if os.path.isfile(os.path.join(sp_path, 'parmeters.txt')):
        # if it already exist make sure that the parameters are the same
        parameter_text = 'model={}, max_seq_len={}\n'.format(str(model_type), max_seq_len)
        parameter_text += ', '.join(['{}={}'.format(arg, spin_text_args[arg]) for arg in spin_text_args])
        with open(os.path.join(sp_path, 'parmeters.txt'), encoding='utf-8') as file:
            assert file.read() == parameter_text, f"The parameters from in {os.path.join(sp_path, 'parmeters.txt')} do not line up with the used parametes!"
    else:
        # create a file to store the parameters of the creation of the text
        parameter_text = 'model={}, max_seq_len={}\n'.format(str(model_type), max_seq_len)
        parameter_text += ', '.join(['{}={}'.format(arg, spin_text_args[arg]) for arg in spin_text_args])
        with open(os.path.join(sp_path, 'parmeters.txt'), 'w', encoding='utf-8', newline='\n') as file:
            file.write(parameter_text)

    # getting the first N names of the original text files which were not paraphrased already
    all_og_files = [f for f in os.listdir(og_path) if os.path.isfile(os.path.join(og_path, f))]
    sp_path_texts = os.path.join(sp_path, 'text')
    all_spun_files = [f.replace('SPUN', 'ORIG') for f in os.listdir(sp_path_texts) if os.path.isfile(os.path.join(sp_path_texts, f))]
    # include excluded files into the processed files
    if os.path.isfile(os.path.join(sp_path, 'excluded.txt')):
        with open(os.path.join(sp_path, 'excluded.txt'), encoding='utf-8') as file:
            all_spun_files.extend(file.read().split('\n')[:-1])
    og_files = np.setdiff1d(all_og_files, all_spun_files)
    if len(og_files) == 0:
        print('All files were already paraphrased!!!')
        exit()
    elif len(og_files) > N:
        og_files = og_files[:N]

    # check if the dataset DataFrame already exist
    if os.path.isfile(os.path.join(sp_path, 'dataset.pkl')):
        # if yes then load it
        df_dataset = pd.read_pickle(os.path.join(sp_path, 'dataset.pkl'))
    else:
        # if not set it to None
        df_dataset = None
    for og_file in tqdm(og_files):
        with open(os.path.join(og_path, og_file), encoding='utf-8') as file:
            originalText = file.read()
        try:
            spun_text, df = spin_text(originalText, tokenizer, lm, **spin_text_args)

            df_dataset = add_to_df_dataset(df, df_dataset)
            spun_file = og_file.replace('ORIG', 'SPUN')
            with open(os.path.join(sp_path, *['text', spun_file]), 'w', encoding='utf-8', newline='\n') as file:
                file.write(spun_text)
            spun_df_file = spun_file.replace('txt', 'pkl')
            df.to_pickle(os.path.join(sp_path, *['df', spun_df_file]))
        except AssertionError:
            # store files which cause an AssertionError in spin_text in excluded.txt
            with open(os.path.join(sp_path, 'excluded.txt'), 'a', encoding='utf-8', newline='\n') as file:
                file.write(og_file + "\n")

    # check if the hole dataset was paraphrased
    all_og_files = [f for f in os.listdir(og_path) if os.path.isfile(os.path.join(og_path, f))]
    all_spun_files = [f.replace('SPUN', 'ORIG') for f in os.listdir(sp_path_texts) if os.path.isfile(os.path.join(sp_path_texts, f))]
    # include excluded files into the processed files
    if os.path.isfile(os.path.join(sp_path, 'excluded.txt')):
        with open(os.path.join(sp_path, 'excluded.txt'), encoding='utf-8') as file:
            all_spun_files.extend(file.read().split('\n')[:-1])
    if len(all_og_files) == len(all_spun_files):
        # then create the avg out of the sums in the dataset DataFrame
        df_dataset = df_dataset.fillna({'chosen': 0}).astype({'chosen': 'int64'})
        df_dataset = df_dataset.fillna({'suggested': 0}).astype({'suggested': 'int64'})
        df_dataset['avg_sug_score'] = df_dataset['sum_sug_score']/df_dataset['suggested']
        df_dataset['avg_cho_score'] = df_dataset['sum_cho_score']/df_dataset['chosen']
        df_dataset = df_dataset[['suggested', 'avg_sug_score', 'chosen', 'avg_cho_score']]
        print('The hole Dataset was paraphrased!')

    # store the dataset DataFrame
    df_dataset.to_pickle(os.path.join(sp_path, 'dataset.pkl'))
