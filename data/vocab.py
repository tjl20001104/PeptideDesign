import torch.nn as nn
import pandas as pd
from torchtext.vocab import vocab
from collections import Counter, OrderedDict
import dill
import numpy as np


def construct_vocab(text_vocab_path,sentences_cut=None,data_path=''):
    if sentences_cut is None:
        df = pd.read_csv(data_path) #读取文件，记住取名不要用中文
        sentences = df['text']
        sentences = sentences.tolist()
        sentences_cut = [line.split() for line in sentences]
    all_entry = []
    for line in sentences_cut:
        all_entry.extend(line)
    counter = Counter(all_entry)  # 统计计数
    sorted_by_freq_tuples = sorted(counter.items(), key=lambda x: x[1], reverse=True)  # 构造成可接受的格式：[(单词,num), ...]
    ordered_dict = OrderedDict(sorted_by_freq_tuples)
    # 开始构造 vocab
    my_vocab = vocab(ordered_dict, specials=['<unk>', '<pad>', '<start>', '<eos>'])  # 单词转token，specials里是特殊字符，可以为空
    my_vocab.set_default_index(0)
    # 保存 vocab
    with open(text_vocab_path, 'wb')as f:
        dill.dump(my_vocab, f)
    
    return my_vocab


# data_path = "data/all_amp.csv"
# text_vocab_path = "vocab"
# construct_vocab(data_path=data_path, text_vocab_path=text_vocab_path)

# text_vocab_path = "vocab"
# with open(text_vocab_path, 'rb')as f:
#     my_vocab = dill.load(f)

# vocab_emb = nn.Embedding(len(my_vocab), 150, 1)
# breakpoint()