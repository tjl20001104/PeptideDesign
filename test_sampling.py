import logging
import os
from os.path import join as pjoin
import torch
import numpy as np
import random
import argparse
import dill
import pandas as pd
from sklearn.metrics import roc_auc_score as AUC
from sklearn.metrics import accuracy_score
from fitter import Fitter, get_distributions
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

from data import dataset
from models.model import RNN_WAE
from models.classifier import build_classifier
from train_classifier import train_classifier,test_classifier
import tb_json_logger
import utils
import cfg
from sample_pipeline import compute_modlamp
from utils import decode_from_z

device = torch.device("cuda:0" if torch.cuda.is_available() and not cfg.ignore_gpu else "cpu")
cfg._update_cfg()

with open(cfg.vocab_path, 'rb')as f:
    my_vocab = dill.load(f)

# DATA
classifier_train_dataset = dataset.construct_dataset(
    cfg.dataset.train_data_path,
    cfg.dataset.current_label,
    cfg.dataset.vocab_path,
    cfg.max_seq_len)

# MODEL
model = RNN_WAE(n_vocab=len(my_vocab), max_seq_len=cfg.max_seq_len,
                **cfg.model).to(device)

for name_classifier in cfg.all_classifier:
    if not hasattr(model, name_classifier):
        classifier = build_classifier('dnn', cfg.model.z_dim, **cfg.classifier.C_args).to(device)
        model.add_module(name_classifier, classifier)

samples_z_untrained = model.sample_z_prior(4000)
samples_untrained = decode_from_z(samples_z_untrained, model, classifier_train_dataset)
df_untrained = pd.DataFrame({'peptide': samples_untrained,'z': [tuple(z.tolist()) for z in samples_z_untrained]})

if cfg.loadpath:
    if os.path.exists(cfg.loadpath):
        model.load_state_dict(torch.load(cfg.loadpath), strict=False)
        print('Loaded model from ' + cfg.loadpath)
    else:
        print('Train new model')

model.eval()
samples_z = model.sample_z_prior(4000)
samples = decode_from_z(samples_z, model, classifier_train_dataset)
df = pd.DataFrame({'peptide': samples,'z': [tuple(z.tolist()) for z in samples_z]})

df_untrained = df_untrained[~df_untrained['peptide'].isin([''])]
df_untrained = compute_modlamp(df_untrained)
df = df[~df['peptide'].isin([''])]
df = compute_modlamp(df)

df_untrained.to_csv('./z_result/untrained_samples.csv')
df.to_csv('./z_result/trained_samples.csv')
breakpoint()

df = utils.clf_pred(cfg.dataset.label_name, df, model, classifier_train_dataset)

df = df[~df['peptide'].isin([''])]
df = compute_modlamp(df)
df.to_csv('./z_result/raw_samples.csv')

# df_untrained = df_untrained[~df_untrained['peptide'].isin([''])]
# df_untrained = compute_modlamp(df_untrained)
# df_untrained.to_csv('./raw_samples.csv')

# instability = df['instability_index']
# instability_untrained = df_untrained['instability_index']

# data = [instability, instability_untrained]
# positions = [1,2]

# fig,ax=plt.subplots()
# ax.violinplot(data,positions)
# plt.show()

df_select = df[(df['instability_index']<=40)&(df['tox']<0.01)&(df['amp']>0.8)]
df_select.to_csv('./z_result/selected_samples.csv')
# breakpoint()

idx = 0
with open('./z_result/seq.fasta', 'w') as f:
    for seq in df_select['peptide']:
        seq_fasta = seq.replace(" ","")
        if len(seq_fasta)<5:
            continue
        idx += 1
        f.write('>{}\n'.format(idx))
        f.write(seq_fasta+'\n')
# breakpoint()