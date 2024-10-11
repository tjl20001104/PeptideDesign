import sys
sys.path.append('../')

import pandas as pd
from torchtext.vocab import vocab
from collections import Counter, OrderedDict
from torch.utils.data import Dataset, DataLoader
from torchtext.transforms import VocabTransform,ToTensor,AddToken
import torch.nn as nn
import torch
import os
import dill
import numpy as np
from data import dataset
from data import vocab
from models.mutils import UNK_IDX, PAD_IDX, START_IDX, EOS_IDX

class AmpDataSet(Dataset):
    def __init__(self, text_list, label_list, text_vocab_path, fixed_length):
        """
        使用新版API的一个简单的TextDataSet
        :param text_list: 语料的全部句子
        """
        if os.path.exists(text_vocab_path): #如果已有创建好的vocab
            print('加载已创建的词汇表...')
            with open(text_vocab_path, 'rb')as f:
                my_vocab = dill.load(f)
        else: # 如果没有vocab,开始构造vocab
            print('本地没有发现词汇表,新建词汇表...')
            my_vocab = vocab.construct_vocab(text_vocab_path, sentences_cut=text_list)
        vocab_transform = VocabTransform(my_vocab)
        start_token = AddToken(token=my_vocab['<start>'],begin=True)
        end_token = AddToken(token=my_vocab['<eos>'],begin=False)
        # 开始构造DataSet
        self.len_list = [len(line) for line in text_list]
        self.text_list = text_list  # 原始文本
        self.label_list = label_list
        self.vocab = my_vocab
        self.vocab_transform = vocab_transform
        self.start_token = start_token
        self.end_token = end_token
        self.fixed_length = fixed_length
        self._len = len(text_list)  # 文本量
        self.n_vocab = len(my_vocab)

    def __getitem__(self, id_index):  # 每次循环的时候返回的值
        sentence = self.text_list[id_index]
        word_ids = self.vocab_transform(sentence)
        word_ids = self.end_token(self.start_token(word_ids))
        if self.fixed_length + 2 > len(word_ids):
            word_ids = word_ids + [self.vocab['<pad>']]*(self.fixed_length-len(word_ids)+2)
        word_tensor = np.array(word_ids)
        word_tensor = torch.from_numpy(word_tensor)
        label_tensor = self.label_list[id_index]
        label_tensor = np.array(label_tensor)
        label_tensor = torch.from_numpy(label_tensor)
        return word_tensor, label_tensor, self.len_list[id_index]+2

    def __len__(self):
        return self._len

    def idx2sentences(self, idxs, print_special_tokens=True):
        """ recursively descend into n-dim tensor or list and return same nesting """
        if not isinstance(idxs[0], list) and (isinstance(idxs[0], (int, float)) or idxs[0].dim() == 0):
            # 1D, no more nesting
            return self.idx2sentence(idxs, print_special_tokens)
        else:
            return [self.idx2sentences(s, print_special_tokens) for s in idxs]

    def idx2sentence(self, idxs, print_special_tokens=True):
        assert isinstance(idxs, list) or idxs.dim() == 1, 'expecting single sentence here'
        if not print_special_tokens:
            idxs = [i for i in idxs if i not in [UNK_IDX, PAD_IDX, START_IDX, EOS_IDX]]  # filter out
        res = ' '.join([self.vocab.get_itos()[i] for i in idxs])
        return res

    def sentence2idx(self, sentence):
        assert isinstance(sentence, list), 'expecting single sentence here'
        word_ids = self.vocab_transform(sentence)
        if len(word_ids) > 50:
            word_ids = word_ids[:50]
        word_ids = self.end_token(self.start_token(word_ids))
        if self.fixed_length + 2 > len(word_ids):
            word_ids = word_ids + [self.vocab['<pad>']]*(self.fixed_length-len(word_ids)+2)
        word_tensor = np.array(word_ids)
        word_tensor = torch.from_numpy(word_tensor)
        return word_tensor

    def sentences2idx(self, sentences):
        if not isinstance(sentences[0], list):
            # 1D, no more nesting
            return self.sentence2idx(sentences)
        else:
            return [self.sentence2idx(s) for s in sentences]

def construct_dataset(data_set_path,label_name,vocab_path,fixed_length):
    df = pd.read_csv(data_set_path) #读取文件，记住取名不要用中文
    sentences = df['text']
    sentences = sentences.tolist()
    sentences_cut = [line.split() for line in sentences]
    if label_name != None and label_name in df:
        labels = df[label_name].tolist()
        label_list = []
        for item in labels:
            if 'neg' in item:
                label_list.append([0])
            elif 'pos' in item:
                label_list.append([1.])
        if len(label_list) != len(sentences_cut):
            print('label长度与text不同!!!')
            raise AssertionError
    else:
        label_list = [[0] for item in sentences_cut]
    text_dataset = dataset.AmpDataSet(sentences_cut,label_list,text_vocab_path=vocab_path,fixed_length=fixed_length)
    return text_dataset


def main():
    data_path = "all_amp.csv"
    vocab_path = "vocab"
    dataset = construct_dataset(data_path,'amp',vocab_path,50) # 'amp' or 'tox'
    # df = pd.read_csv(data_path) #读取文件，记住取名不要用中文
    # sentences = df['text']
    # sentences = sentences.tolist()
    # sentences_cut = [line.split() for line in sentences]
    # text_dataset = AmpDataSet(sentences_cut,text_vocab_path=vocab_path,fixed_length=50)  # 构造 DataSet
    data_loader = DataLoader(dataset, batch_size=10)  # 将DataSet封装成DataLoader
    for word_tensor, label_tensor, len_sentence in data_loader:
        print("====================================")
        print("原句是：", word_tensor)
        print("对应的tensor：", label_tensor)
        breakpoint()


if __name__ == '__main__':
    main()