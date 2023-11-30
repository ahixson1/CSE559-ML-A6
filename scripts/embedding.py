import pandas as pd
from pathlib import Path
import torch
from allennlp.commands.elmo import ElmoEmbedder

print('START')

model_dir = Path('./MLProject/catELMo')
weights = model_dir/'embedders'/'weights.hdf5'
options = model_dir/'embedders'/'options.json'

embedder = ElmoEmbedder(options,weights,cuda_device=0) # cuda_device=-1 for CPU

def catELMo_embedding(x):
    return torch.tensor(embedder.embed_sentences(list(x))).sum(dim=0).mean(dim=0).tolist()


dat = pd.read_csv('./MLProject/catELMo/datasets/TCREpitopePairs.csv')
print('HERE')

dat['tcr_embeds'] = None
dat['epi_embeds'] = None

dat['epi_embeds'] = dat[['epi']].applymap(lambda x: catELMo_embedding(x))['epi']
dat['tcr_embeds'] = dat[['tcr']].applymap(lambda x: catELMo_embedding(x))['tcr']

dat.to_pickle("./MLProject/catELMo_base_data.pkl")
