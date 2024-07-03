import pandas as pd
import json
from tqdm import tqdm
import re
df = pd.read_csv("train1.csv")
words = {}
index = 0

for _ in tqdm(df.index):
    for _ in df.iloc[_]['target'].split(" "):
        _ = cleanString = re.sub(r'[^A-Za-z0-9\\]+','', _ ).lower()
        if words.get(_) != None:  
            words[_][0] += 1
            continue
        words[_] = [1,index]     
        index +=1
with open('./word_count.json','w') as f:
  json.dump(words, f, ensure_ascii=False, indent=4)
  