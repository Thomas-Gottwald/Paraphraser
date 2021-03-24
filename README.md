# Neural Obfuscation Generator -NOGo

This repository contains the implementation of a transformer based paraphrasing tool, which was created for a student project in Applied Neural Language Processing.
The goal of the project was to implement a transformer based paraphraser and then use it to spin three datasets consisting of wikipedia, arxiv and thesis paragraphs, also
investigated in the paper [Detecting Machine-obfuscated Plagiarism](https://www.gipp.com/wp-content/papercite-data/pdf/foltynek2020.pdf) (the spun datasets can be accessed [here]()).<br>
The basic idea for paraphrasing texts used here is to replace words of the original text with different ones suggested by a masking language model
(the new tokens are chosen by taking the one with highest score that is not the old token or the old token with different casing).

## Requirements

For the project the open source python distribution of [anaconda](https://www.anaconda.com/products/individual) was used.

To set up the environment using anaconda

```setup
conda env create -n paraphraser -f environment_paraphraser.yml
```

Activate the environment

```setup
conda activate paraphraser
```

Deactivate an environment

```setup
conda deactivate
```

### Use of Dataset

Datasets should be inserted into the repository into a folder named "data" so that the files with the original texts are stored at Paraphraser/data/Dataset/ogUTF-8 with Dataset being the
name of the dataset. The original paragraphs should all be in separate files in the utf-8 format with linux line endings ("\n") and also should contain "ORIG" in there file names.<br>
Samples will be stored in Paraphraser/data/sample and when paraphrasing hole dataset the spun files are stored in Paraphraser/data/Dataset/sp(Model,mask_prob)/text.

## Spin Paragraphs

To apply the paraphraser on a single text.

```python
from Paraphraser.paraphraser import init_model, spin_text, Model

text = "Some text that should be paraphrased."

model_type = Model.ROBERTA
max_seq_len = 512
mask_prob = 0.5
seed = 646845

tokenizer, lm = init_model(model_type, max_seq_len, enable_cuda=True)

spun_text, df = spin_text(text, tokenizer, mask_prob, seed=seed)
```

- model_type refers to the masking model used, which can be set via the Enum Model or a string to load the model from [hugging face transformers](https://huggingface.co/models) 
- max_seq_len is the maximum length of tokens that can be inserted into the language model.
  Longer sequences are split at a new sentence. 
- mask_prob sets the amount of words that are replaced by the paraphraser. Only real words are replaced and named entities are also excluded.
- seed can be set as an integer to alow for recreation of the same results
- tokenizer is the tokenizer of the masking model
- lm is the masking model. If enable_cuda is True and cuda is available the model is transferred to the GPU
- spun_text is the resulting paraphrased text
- df is a pandas DataFrame with information for the replaced tokens it contains the index, token, POS (part of speech) of the original tokens and the top-k
  (k can be set in spin_text default 5) suggested tokens with there score and POS.

## Sample Spun texts

Spin some sample texts out of a dataset.<br>
The sample will contain one file with the original and spun texts and one file containing the DataFrames of the spun texts for all texts chosen from the datasets.
When disguise_sample is True the information on the order of the texts in the files is stored in an info file.

```python
from Paraphraser.paraphraser import Data, Model
from Paraphraser.sampling import create_sample

seed = 646845
create_sample(
    sample_size=3,
    data=[Data.WIKIPEDIA],
    spin_text_args=[{'mask_prob': 0.5, 'max_prob': 0.1, 'k': 5, 'seed': seed}],
    model_type=Model.ROBERTA,
    disguise_sample=False
)
```

- sample_size determents how many text will be paraphrased
- data is a list of datasets, which can be ether set via the Enum Data or the folder name of the dataset in Paraphraser/data
- spin_text_args is a list of argument dictionaries used for the created spun texts
- model_type refers to the masking model used
- disguise_sample determents wether the information if a text was spun or not is stored in the same file as the texts or in a separate log file

## Spin Datasets

When paraphrasing hole datasets it is stored at Paraphraser/data/sp(Model,mp), where Model is a reference to the used language model and mp is the used mask_prob.
The spun texts are then stored in the subfolder "text" and the pickle files for the DataFrames of the spun texts are stored in the folder "df".
When some file can not be paraphrased for instance its text contains no words that can be paraphrased its name is put into the file excluded.txt.
More precise information of the parameters used is stored in parameters.txt.
And dataset.pkl contains a DataFrame with information on how often for a certain original POS other POS were suggested or chosen and which average scores they had.

```python
from Paraphraser.paraphraser import Data, Model
from Paraphraser.file_paraphraser import paraphrase_datasets

data = Data.WIKIPEDIA
N = 10000
model_type = Model.ROBERTA
max_seq_len = 512
mask_prob = 0.5
max_prob = 0.1
k = 5
spin_text_args = {'mask_prob': mask_prob, 'max_prob': max_prob, 'k': k}

paraphrase_dataset(data, N, model_type, max_seq_len, spin_text_args)
```

- data refers to the dataset that should be spun ether via the Enum Data or the folder name of the dataset in Paraphraser/data
- N is the amount of files that should be spun (only files are spun which where not spun already and there name is not in excluded.txt).
  It is possible to only spin a part of a datasets and continue later. 
- spin_text_args is a dictionary containing the parameters for the paraphraser

## Resources

The in the Project spun datasets (using the language models RoBERTa and BART) can be found on zenodo [here]().