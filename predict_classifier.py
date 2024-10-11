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

with open('candidate_sequences.txt','r') as f:
    data_txt = f.read()
data_list = data_txt.split('\n')
seq_list = [' '.join(list(data)) for data in data_list]
df = pd.DataFrame({'peptide': seq_list})

model.eval()
# breakpoint()
# df = compute_modlamp(df)
df = utils.clf_pred(cfg.dataset.label_name, df, model, classifier_train_dataset)
df.to_csv('res.csv')