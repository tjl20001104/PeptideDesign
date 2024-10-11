import pandas as pd
import numpy as np
import torch
import os
import dill
from matplotlib import pyplot as plt
from models.model import RNN_WAE
import cfg
from data import dataset
from models.classifier import build_classifier
from train_classifier import train_classifier,test_classifier
from sample_pipeline import compute_modlamp

cfg._update_cfg()
device = torch.device("cuda:0" if torch.cuda.is_available() and not cfg.ignore_gpu else "cpu")

with open(cfg.vocab_path, 'rb')as f:
    my_vocab = dill.load(f)

# DATA
classifier_train_dataset = dataset.construct_dataset(
    cfg.dataset.train_data_path,
    cfg.dataset.current_label,
    cfg.dataset.vocab_path,
    cfg.max_seq_len)

model = RNN_WAE(n_vocab=len(my_vocab), max_seq_len=cfg.max_seq_len,
                **cfg.model).to(device)

if cfg.loadpath:
    if os.path.exists(cfg.loadpath):
        model.load_state_dict(torch.load(cfg.loadpath), strict=False)
        print('Loaded model from ' + cfg.loadpath)
    else:
        print('Train new model')

def decode_from_z(z, model, dataset):
    sall = []
    for zchunk in torch.split(z, 1000):
        s, _ = model.generate_sentences(zchunk.size(0),
                zchunk,
                sample_mode='categorical')
        sall += s
    return dataset.idx2sentences(sall, print_special_tokens=False)

samples_z = model.sample_z_prior(4000)
samples = decode_from_z(samples_z, model, classifier_train_dataset)
df = pd.DataFrame({'peptide': samples,'z': [tuple(z.tolist()) for z in samples_z]})
df = df[~df['peptide'].isin([''])]
df = compute_modlamp(df)
breakpoint()