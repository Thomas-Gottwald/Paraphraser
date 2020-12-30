from os import listdir
from os.path import isfile, join
# arxiv
# ogpath = r"./Paraphraser/data/arxiv/og"
# utf8path = r"./Paraphraser/data/arxiv/ogUTF-8"
# thesis
# ogpath = r"./Paraphraser/data/thesis/og"
# utf8path = r"./Paraphraser/data/thesis/ogUTF-8"
# wikipedia
ogpath = r"./Paraphraser/data/wikipedia/og"
utf8path = r"./Paraphraser/data/wikipedia/ogUTF-8"

ogfiles = [f for f in listdir(ogpath) if isfile(join(ogpath, f))]


for i in range(len(ogfiles)):
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

for i in range(len(ogfiles)):
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