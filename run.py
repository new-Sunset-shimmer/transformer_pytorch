#external module
import torch
import copy
import os 
import sys
import pandas as pd
import json
from collections import OrderedDict
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
#internal module
from config.config import ex
from model.transform import Transform
from data.loader import preprocess
from utils.utils import preprocess as pp
from utils.utils import MyDataset
torch.manual_seed(11)
@ex.automain
def main(_config):
    _config = copy.deepcopy(_config)
    print("------------------------------------------------------------")
    for key, val in _config.items():
        print(f"{key} = {val}")
    print("------------------------------------------------------------")
    
    os.environ["PATH"] += os.pathsep + _config['run_enviroment']
    data_path = f"data/{_config['file_name']}"
    os.chdir(_config["run_enviroment"])
    if not os.path.isfile(data_path+".csv"):
        preprocess(_config["file_name"],"0000")
    else: 
        print("------------------------------------------------------------")
        print("exist")
        print("------------------------------------------------------------")
    data = pd.read_csv(data_path+".csv")
    with open("./data/vocab.json",'r') as f:
        vocab = OrderedDict(json.load(f))
    with open("./data/vocab_label.json",'r') as f:
        vocab_label = OrderedDict(json.load(f))
    max_vocab = vocab[tuple(vocab.keys())[-1]][1]
    max_vocab_label = vocab_label[tuple(vocab_label.keys())[-1]][1]
    model = Transform(vocab,vocab_label,_config['label_length'],_config['context_length'],max_vocab,max_vocab_label)
    # loss_f = torch.nn.CrossEntropyLoss(reduction='mean')
    loss_f = torch.nn.CrossEntropyLoss(reduction='mean',ignore_index=-1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    dataset = MyDataset(data,vocab,vocab_label,_config['context_length'],_config['label_length'],max_vocab,max_vocab_label)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=_config['batch'], shuffle=True)
    dataloader_test = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    model.train()
    pads = max_vocab_label+1
    writer = SummaryWriter("model/16")
    id = 0
    for i in tqdm(range(_config['epoch'])):
        for string,tag in dataloader:
            # print("******************* input *******************")
            x = model.forward(string,tag)
            x.requires_grad_(True)
            tag_mask = tag != pads
            x = x.where(tag_mask, torch.tensor(-1))
            tag = tag.where(tag_mask,torch.tensor(-1))
            # x = x[tag_mask]
            # tag = tag[tag_mask]
            # tag = torch.nn.functional.one_hot(tag,num_classes = max_vocab_label + 3)
            loss = loss_f(x,tag.to(torch.float))
            writer.add_scalar("Loss/train/epochs",loss, id)
            id += 1
            loss.backward()
            optimizer.step()
            # optimizer.zero_grad()
    torch.save(model, "test.pth")
    model.eval()
    model = torch.load("test.pth")
    for string,tag in dataloader_test:
        print(model.test(string,tag,dataset.vocabl,dataset.pipeline_label))

