import pandas as pd
import json
from tqdm import tqdm
import re
import torch
import duckdb
from torchtext.data import get_tokenizer
from torchtext.vocab import vocab
from collections import OrderedDict
def Par2Csv(file,name):
    test = pd.read_parquet(f'./data/{file}.parquet', engine='pyarrow')
    lens = len(test)
    _ = 0
    test.to_csv(f"./data/{name}.csv",index=False)
def Vocab_builder(file,name,target):
    df = pd.read_csv(f"./data/{file}.csv")
    words = {}
    index = 0
    for _ in tqdm(df.index):
        for _ in df.iloc[_][target].split(" "):
            _ = cleanString = re.sub(r'[^A-Za-z0-9\\]+','', _ ).lower()
            if words.get(_) != None:  
                words[_][0] += 1
                continue
            words[_] = [1,index]     
            index +=1
    with open(f'./data/{name}.json','w') as f:
        json.dump(words, f, ensure_ascii=False, indent=4)
def preprocess(str,tag,vocab,vocabl,vocab_max,vocabl_max):
    # words = torch.zeros((vocab_max,vocab_max),dtype=torch.int)
    # words_tag = torch.zeros((vocabl_max,vocabl_max),dtype=torch.int)
    words = []
    words_tag = []
    for _ in str.split(" "):
        words.append(vocab[re.sub(r'[^A-Za-z0-9\\]+','', _ ).lower()][1])
    for _ in tag.split(" "):
        words_tag.append(vocabl[re.sub(r'[^A-Za-z0-9\\]+','', _ ).lower()][1])
    words = torch.tensor(words)
    words_tag = torch.tensor(words_tag)
    return words,words_tag
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, vocabs, vocabl,len,lenl,max_voc,max_lab):
        self.vocab_max = len
        self.vocabl_max = lenl
        self.dataframe = dataframe
        self.tokenizer = get_tokenizer("basic_english")
        self.vocab = vocab(OrderedDict([(token, vocabs[token][0]) for token in vocabs]))
        self.vocab.set_default_index(max_voc)
        self.vocabl = vocab(OrderedDict([(token, vocabl[token][0]) for token in vocabl]))
        self.vocabl.append_token("<unk>")
        self.vocabl.append_token("<pad>")
        self.vocabl.set_default_index(max_lab)
        self.pipeline = lambda x : self.vocab(self.tokenizer_helper(x))
        self.pipeline_label = lambda x : self.vocabl(self.tokenizer_helper(x))
        self.pad = int(max_voc+1)
        self.padl = int(max_lab+1)
        print(self.vocabl.lookup_token(max_lab+1))
    def tokenizer_helper(self,x):
        y = self.tokenizer(re.sub(r'[^A-Za-z0-9\\]+',' ', x ))
        return y
    def __len__(self):
        return len(self.dataframe["concept_set_idx"])
    def label_token(self,index):
        words = self.pipeline(self.dataframe.iloc[index]['target'])
        padding = self.vocab_max - len(words)
        words = torch.nn.functional.pad(torch.tensor(words), (0, padding),value = self.pad)
        return words
    def target_toekn(self,index):
        words_label = self.pipeline_label(self.dataframe.iloc[index]['concepts'])
        padding = self.vocabl_max - len(words_label)
        words_label = torch.nn.functional.pad(torch.tensor(words_label), (0, padding),value = self.padl)
        return words_label
    def __getitem__(self, index):
        # words = []
        # words_label = []
        # for _ in self.dataframe[index]['target'].split(" "):
        #     words.append(self.vocab[re.sub(r'[^A-Za-z0-9\\]+','', _ ).lower()][1])
        # for _ in self.dataframe[index]['concepts'].split(" "):
        #     words_label.append(self.vocab[re.sub(r'[^A-Za-z0-9\\]+','', _ ).lower()][1])
        words = self.label_token(index)
        words_label = self.target_toekn(index)
        return words,words_label