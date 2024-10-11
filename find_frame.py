import pandas as pd
import numpy as np
import torch
import os
import dill
import difflib
from matplotlib import pyplot as plt
from models.model import RNN_WAE
import cfg
from data import dataset
from models.classifier import build_classifier
from train_classifier import train_classifier,test_classifier
from sample_pipeline import compute_modlamp
import utils


# generate config
cfg._update_cfg()
max_seq_len = 200

# torch-related setup from cfg.
device = torch.device("cuda:0" if torch.cuda.is_available() and not cfg.ignore_gpu else "cpu")

with open(cfg.vocab_path, 'rb')as f:
    my_vocab = dill.load(f)

# MODEL
model = RNN_WAE(n_vocab=len(my_vocab), max_seq_len=max_seq_len,
                **cfg.model).to(device)

for name_classifier in cfg.all_classifier:
    if not hasattr(model, name_classifier):
        classifier = build_classifier('dnn', cfg.model.z_dim, **cfg.classifier.C_args).to(device)
        model.add_module(name_classifier, classifier)

if cfg.loadpath:
    if os.path.exists(cfg.loadpath):
        model.load_state_dict(torch.load(cfg.loadpath), strict=False)
        print('Loaded model from ' + cfg.loadpath)
    else:
        print('Train new model')

# DATA
classifier_train_dataset = dataset.construct_dataset(
    cfg.dataset.train_data_path,
    cfg.dataset.current_label,
    cfg.dataset.vocab_path,
    max_seq_len)

df = pd.read_csv('selected.csv', index_col=0)
Population = df['peptide'].tolist()
Population = [p.replace(" ", "") for p in Population]
Population = [[Population[i], set([(i, len(Population[i]))])] for i in range(len(Population))]

def find_overlap(t1, t2):
    s1, s2 = t1[0], t2[0]
    set1, set2 = t1[1], t2[1]
    s = difflib.SequenceMatcher(None, s1, s2)
    pos_a, pos_b, size = s.find_longest_match(0, len(s1), 0, len(s2)) 
    return [s1[pos_a:pos_a+size], set1|set2]

def check_frame_valid(t1):
    seq, sets = t1
    len_overlap = len(seq)
    if len_overlap > 10:
        return 1
    for s in sets:
        if len_overlap / s[1] > 0.9:
            return 1
    return 0
    
def merge_identical(overlaps):
    seq = [s[0] for s in overlaps]
    identical_seq = set(seq)
    identical_seq = [[s, set()] for s in identical_seq]
    for t1 in overlaps:
        for t2 in identical_seq:
            if t1[0] == t2[0]:
                t2[1] = t2[1] | t1[1]
    return identical_seq

def recursive_search(Population):
    good_backbone = []
    while True:
        overlaps = []
        for i in range(len(Population)):
            for j in range(i+1, len(Population)):
                overlap = find_overlap(Population[i], Population[j])
                if check_frame_valid(overlap):
                    overlaps.append(overlap)
        if len(overlaps) == 0:
            break
        overlaps = merge_identical(overlaps)
        Population = overlaps
        for t in Population:
            if len(t[1]) >= 5:
                good_backbone.append(t)
        good_backbone = merge_identical(good_backbone)
    return good_backbone

def match_parent(backbones, Populations):
    result = []
    for b in backbones:
        pair = {}
        pair['backbone'] = b[0]
        parent = []
        for s in b[1]:
            seq = Populations[s[0]][0]
            if len(seq) == s[1]:
                parent.append(seq)
        pair['parents'] = ' \n '.join(parent)
        result.append(pair)
    return result

good_backbone = recursive_search(Population)
result = match_parent(good_backbone, Population)
df = pd.DataFrame(result)
df.to_csv('result_backbone.csv')