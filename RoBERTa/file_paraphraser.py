import paraphrase_pipeline as ppipe
from transformers import pipeline
from transformers.pipelines import FillMaskPipeline
from transformers import AutoTokenizer, AutoModelForMaskedLM
import os


tokenizer = AutoTokenizer.from_pretrained("roberta-large")
model = AutoModelForMaskedLM.from_pretrained("roberta-large")
model.resize_token_embeddings(len(tokenizer))
unmasker = FillMaskPipeline(model=model, tokenizer=tokenizer, device=0)
# unmasker = pipeline('fill-mask', model='roberta-large')
paraphraser = ppipe.ParaphrasePipeline(unmasker, input_window_size=200)

# thesis
ogpath = r"./Paraphraser/data/thesis/ogUTF-8"
sppath = r"./Paraphraser/data/thesis"

# arxiv
# ogpath = r"./Paraphraser/data/arxiv/ogUTF-8"
# sppath = r"./Paraphraser/data/arxiv"

# wikipedia
# ogpath = r"./Paraphraser/data/wikipedia/ogUTF-8"
# sppath = r"./Paraphraser/data/wikipedia/"

ogfiles = [f for f in os.listdir(ogpath) if os.path.isfile(os.path.join(ogpath, f))]

mask_pc = 20 # 10%, 20%, 30%
sp_dir = "sp({}%)".format(mask_pc)

sppath = os.path.join(sppath, sp_dir)
if not os.path.exists(sppath):
    os.makedirs(sppath)

N = len(ogfiles)
log_val = 100
for i in range(N):
    if i % log_val == 0:
        print('[{} / {}]'.format(i+1, N))
    with open(os.path.join(ogpath, ogfiles[i]), encoding='utf-8') as file:
        originalText = file.read()
    spun_text = paraphraser.parapherase(originalText, mask=mask_pc/100, range_replace=(1, 4))
    spunfile = ogfiles[i].replace('ORIG', 'SPUN')
    f = open(os.path.join(sppath, spunfile), 'w', encoding='utf-8', newline='\n')
    f.write(spun_text)
    f.close()

print('Finish!!!')