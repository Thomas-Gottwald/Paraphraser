from os import listdir
from os.path import isfile, join
import os
import platform
from tqdm import tqdm

def get_local_path():
    """
    Gives the path form the environment to the Paraphraser folder
    (in the form "./path/to/environment/")
    """
    cwd = os.getcwd()
    if platform.system() == 'Windows':
        cwd = cwd[0].lower() + cwd[1:]
    file_name = 'changeEncoding.py'
    path = os.path.realpath(__file__)

    path = path.replace(cwd, '.', 1)
    path = path.replace(file_name, '')

    return path

# arxiv
# dataset = 'arxiv'
# thesis
# dataset = 'thesis'
# wikipedia
dataset = 'wikipedia'

ogpath = os.path.join(get_local_path(), *['data', dataset, 'og'])
utf8path = os.path.join(get_local_path(), *['data', dataset, 'ogUTF-8'])

ogfiles = [f for f in listdir(ogpath) if isfile(join(ogpath, f))]

for i in tqdm(range(len(ogfiles))):
    try:
        with open(join(ogpath, ogfiles[i]), 'r', encoding='utf-8') as file:
            text = file.read()
        with open(join(utf8path, ogfiles[i]), 'w', encoding='utf-8') as file:
            file.write(text)
    except:
        try:
            with open(join(ogpath, ogfiles[i]), 'r', encoding='ansi') as file:
                text = file.read()
            with open(join(utf8path, ogfiles[i]), 'w', encoding='utf-8', newline='\n') as file:
                file.write(text)
        except:
            print(f"{ogfiles[i]} no ansi")

print('Coping finished!\nStar now checking is everything was done right.')

for i in tqdm(range(len(ogfiles))):
    try:
        with open(join(ogpath, ogfiles[i]), 'r', encoding='utf-8') as file:
            text = file.read()
    except:
        try:
            with open(join(ogpath, ogfiles[i]), 'r', encoding='ansi') as file:
                text = file.read()
        except:
            print(f"{ogfiles[i]} no ansi")
            break
    with open(join(utf8path, ogfiles[i]), 'r', encoding='utf-8') as file:
            textUTF8 = file.read()

    if text != textUTF8:
        print(f'not equal: {ogfiles[i]}')

print('Finish!!!')