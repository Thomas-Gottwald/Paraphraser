from paraphraser import init_model, spin_text
from getPath import get_local_path
import numpy as np
import pandas as pd
import os
from enum import Enum
from typing import Optional
from tqdm import tqdm

class Data(Enum):
    THESIS = 1
    ARXIV = 2
    WIKIPEDIA = 3

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

def paraphrase_dataset(data: Data, N: int, tokenizer, lm, spin_text_args: dict):
    """
    Paraphrases the dataset referred with data

    Args:
        data: Enum for the Datasets (wikipedia, arxiv, thesis)
        N: The number of files to paraphrase
        tokenizer: The tokenizer of the language model
        lm: The language model used for paraphrasing (the function spin_text)
        spin_text_args: the arguments for the function spin_text besides
            the text to spin, the model tokenizer and the language model
    """
    # setting the path to the data
    path = get_local_path()
    if data == Data.THESIS:
        og_path = os.path.join(path, *['data', 'thesis', 'ogUTF-8'])
        sp_path = os.path.join(path, *['data', 'thesis'])
    elif  data == Data.ARXIV:
        og_path = os.path.join(path, *['data', 'arxiv', 'ogUTF-8'])
        sp_path = os.path.join(path, *['data', 'arxiv'])
    elif data == Data.WIKIPEDIA:
        og_path = os.path.join(path, *['data', 'wikipedia', 'ogUTF-8'])
        sp_path = os.path.join(path, *['data', 'wikipedia'])
    else:
        print('data is not specificied!')
        exit()

    # the folder name for the spun files
    sp_dir = "sp({})".format(spin_text_args['mask_prob'])
    # create the folder if it dose not exist already
    sp_path = os.path.join(sp_path, sp_dir)
    if not os.path.exists(sp_path):
        os.makedirs(sp_path)
        os.makedirs(os.path.join(sp_path, 'text'))
        os.makedirs(os.path.join(sp_path, 'df'))
    
    # check if the parameter text files already exists
    if os.path.isfile(os.path.join(sp_path, 'parmeters.txt')):
        # if it already exist make sure that the parameters are the same
        with open(os.path.join(sp_path, 'parmeters.txt'), encoding='utf-8') as file:
            assert file.read() == ', '.join(['{}={}'.format(
                arg, spin_text_args[arg]) for arg in spin_text_args]
            ), f"The parameters from in {os.path.join(sp_path, 'parmeters.txt')} do not line up with the used parametes!"
    else:
        # create a file to store the parameters of the creation of the text
        with open(os.path.join(sp_path, 'parmeters.txt'), 'w', encoding='utf-8', newline='\n') as file:
            file.write(', '.join(['{}={}'.format(arg, spin_text_args[arg]) for arg in spin_text_args]))

    # getting the first N names of the original text files which were not paraphrased already
    all_og_files = [f for f in os.listdir(og_path) if os.path.isfile(os.path.join(og_path, f))]
    sp_path_texts = os.path.join(sp_path, 'text')
    all_spun_files = [f.replace('SPUN', 'ORIG') for f in os.listdir(sp_path_texts) if os.path.isfile(os.path.join(sp_path_texts, f))]
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
        spun_text, df = spin_text(originalText, tokenizer, lm, **spin_text_args)
        df_dataset = add_to_df_dataset(df, df_dataset)
        spun_file = og_file.replace('ORIG', 'SPUN')
        with open(os.path.join(sp_path, *['text', spun_file]), 'w', encoding='utf-8', newline='\n') as file:
            file.write(spun_text)
        spun_df_file = spun_file.replace('txt', 'pkl')
        df.to_pickle(os.path.join(sp_path, *['df', spun_df_file]))

    # check if the hole dataset was paraphrased
    all_og_files = [f for f in os.listdir(og_path) if os.path.isfile(os.path.join(og_path, f))]
    all_spun_files = [f.replace('SPUN', 'ORIG') for f in os.listdir(sp_path_texts) if os.path.isfile(os.path.join(sp_path_texts, f))]
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

def load_df_dataset(df_path: str):
    # load_df_dataset(os.path.join(get_local_path(), *['data', 'wikipedia', 'sp(0.5)', 'dataset.pkl']))

    df_dataset = pd.read_pickle(df_path)

    print(df_dataset)
    # original POS
    print(df_dataset.reset_index().groupby('og_POS').agg({'chosen': 'sum'}).rename(columns={'chosen': 'count'}))
    # suggested POS
    print(df_dataset.reset_index().groupby('POS').agg({'suggested': 'sum'}))
    # chosen POS
    print(df_dataset.reset_index().groupby('POS').agg({'chosen': 'sum'}))
    # avg chosen score
    if 'avg_cho_score' in df_dataset:
        print((df_dataset['avg_cho_score']*df_dataset['chosen']).sum()/df_dataset['chosen'].sum())


if __name__ == '__main__':
    # witch data will be paraphrased
    data = Data.WIKIPEDIA
    # how many files should be paraphrased
    N = 5000
    # load the language model
    model_name = 'roberta-large'
    max_seq_len = 512
    tokenizer, lm = init_model(model_name, max_seq_len)
    # the parameters for the paraphraser
    mask_prob = 0.5
    max_prob = 0.1
    k = 5
    spin_text_args = {'mask_prob': mask_prob, 'max_prob': mask_prob, 'k': k}

    # spin the dataset
    paraphrase_dataset(data, N, tokenizer, lm, spin_text_args)