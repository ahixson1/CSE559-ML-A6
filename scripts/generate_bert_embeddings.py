import re
import pandas as pd
from tqdm import tqdm
from transformers import TFBertModel,BertModel, BertForPreTraining, BertTokenizer, BertConfig

tokenizer = BertTokenizer.from_pretrained("./base_bert_model/vocab.txt", do_lower_case=False )
model = BertModel.from_pretrained("./base_bert_model/pytorch_model.bin")

def BERT_embedding(x):
    seq = " ".join(x)
    seq = re.sub(r"[UZOB]", "X", seq)
    encoded_input = tokenizer(seq, return_tensors='pt')
    output = model(**encoded_input)
    return output
    
dat = pd.read_csv('./bert_base_embeddings/BindingAffinityPrediction/TCREpitopePairs.csv')
dat['tcr_embeds'] = None
dat['epi_embeds'] = None

for i in tqdm(range(len(dat))):
    dat.epi_embeds[i] = BERT_embedding(dat.epi[i])[0].reshape(-1,1024).mean(dim=0).tolist()
    dat.tcr_embeds[i] = BERT_embedding(dat.tcr[i])[0].reshape(-1,1024).mean(dim=0).tolist()

dat.to_pickle("./bert_base_embeddings/bert_base_embedding.pkl")