from utils.utils import Par2Csv
from utils.utils import Vocab_builder
def preprocess(data,file):
    name = data
    vocab_target = "vocab"
    vocab_concepts = "vocab_label"
    Par2Csv(file,name)
    Vocab_builder(name,vocab_target,"target")
    Vocab_builder(name,vocab_concepts,"concepts")
    