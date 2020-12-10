import paraphrase_pipeline as ppipe
from transformers import pipeline
import os

unmasker = pipeline('fill-mask', model='roberta-large')
paraphraser = ppipe.ParaphrasePipeline(unmasker)

ogpath = r"./Applied Natural Language Processing/Projekt/Paraphraser/data/thesis/ogUTF-8"
sppath = r"./Applied Natural Language Processing/Projekt/Paraphraser/data/thesis"
ogfiles = [f for f in os.listdir(ogpath) if os.path.isfile(os.path.join(ogpath, f))]

mask_pc = 10 # 10%, 20%, 30%
sp_dir = "sp({}%)".format(mask_pc)

sppath = os.path.join(sppath, sp_dir)
if not os.path.exists(sppath):
    os.makedirs(sppath)

N = 1
log_val = 2
for i in range(N):
    if i % log_val == 0:
        print('[{} / {}]'.format(i+1, N))
    with open(os.path.join(ogpath, ogfiles[i]), encoding='utf-8') as file:
        originalText = file.read()
    spun_text = paraphraser.parapherase(originalText, mask=1/mask_pc, range_replace=(1, 4))
    spunfile = ogfiles[i].replace('ORIG', 'SPUN')
    f = open(os.path.join(sppath, spunfile), 'w', encoding='utf-8', newline='\n')
    f.write(spun_text)
    f.close()

print('Finish!!!')