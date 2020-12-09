import paraphrase_pipeline as ppipe
from transformers import pipeline
from os import listdir
from os.path import isfile, join

unmasker = pipeline('fill-mask', model='roberta-large')
paraphraser = ppipe.ParaphrasePipeline(unmasker)

ogpath = r"./Applied Natural Language Processing/Projekt/Paraphraser/data/thesis/og"
sppath = r"./Applied Natural Language Processing/Projekt/Paraphraser/data/thesis/sp"
ogfiles = [f for f in listdir(ogpath) if isfile(join(ogpath, f))]

# print(ogfiles[1])
# with open(join(ogpath, ogfiles[1]), 'r', encoding='latin-1') as file:
#     originalText = file.read()
# print(originalText)

# print(ogfiles[4])
# with open(join(ogpath, ogfiles[4]), 'r', encoding='latin-1') as file:
#     originalText = file.read()
# print(originalText)

# quit()

N = 10
log_val = 2
mask_pc = 10
for i in range(N):
    if i % log_val == 0:
        print('[{} / {}]'.format(i+1, N))
    with open(join(ogpath, ogfiles[i]), encoding='latin-1') as file:
        originalText = file.read()
    spun_text, df = paraphraser.parapherase(originalText, mask=1/mask_pc, range_replace=(1, 4))
    spun_text = spun_text.replace('<s> ', '')
    spun_text = spun_text.replace(' </s>', '')
    spunfile = ogfiles[i].replace('ORIG', 'SPUN({}%)'.format(mask_pc))
    f = open(join(sppath, spunfile), 'w', encoding='utf-8')
    f.write(spun_text)
    f.close()

print('Finish!!!')