import os
import torch
import dill
import numpy as np
import pandas as pd

from data import dataset
import utils
from models.model import RNN_WAE
from models.classifier import build_classifier
from optim.GA_optim import NSGAIII_searching
from sample_pipeline import compute_modlamp
import cfg

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

# samples_z_untrained = model.sample_z_prior(4000)
# samples_untrained = decode_from_z(samples_z_untrained, model, classifier_train_dataset)
# df_untrained = pd.DataFrame({'peptide': samples_untrained,'z': [tuple(z.tolist()) for z in samples_z_untrained]})

if cfg.loadpath:
    if os.path.exists(cfg.loadpath):
        model.load_state_dict(torch.load(cfg.loadpath), strict=False)
        print('Loaded model from ' + cfg.loadpath)
    else:
        print('Train new model')

F = np.load('optim/F.npy')
selected_position = np.load('optim/selected_position.npy')
Population = np.load('optim/Population.npy')
Population_selected = Population[selected_position]
sequences = torch.from_numpy(Population_selected).int().to(model.device)
sequences = classifier_train_dataset.idx2sentences(sequences, print_special_tokens=False)
df = pd.DataFrame({'peptide': sequences})
labels = cfg.dataset.label_name
df = utils.clf_pred(labels, df, model, classifier_train_dataset)
df = compute_modlamp(df)
df.to_csv('optim/selected_samples.csv')

# idx = 0
# with open('optim/seq.fasta', 'w') as f:
#     for seq in df['peptide']:
#         seq_fasta = seq.replace(" ","")
#         if len(seq_fasta)<5:
#             continue
#         idx += 1
#         f.write('>{}\n'.format(idx))
#         f.write(seq_fasta+'\n')