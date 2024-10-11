import logging
import os
from os.path import join as pjoin
import torch
import numpy as np
import random
import argparse
import dill
from sklearn.metrics import roc_auc_score as AUC
from sklearn.metrics import accuracy_score

from data import dataset
from models.model import RNN_WAE
from models.classifier import build_classifier
from train_classifier import train_classifier,test_classifier

import tb_json_logger
import utils
import cfg

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.propagate = False  # do not propagate logs to previously defined root logger (if any).
formatter = logging.Formatter('%(asctime)s - %(levelname)s(%(name)s): %(message)s')
# console
consH = logging.FileHandler('./log.txt','a',delay=False)
consH.setFormatter(formatter)
consH.setLevel(logging.INFO)
logger.addHandler(consH)
# file handler
log = logger

parser = argparse.ArgumentParser(description='PyTorch AMP WAE-GAN and CNN Classifier')
# parser.add_argument('mode', type=str, default=None, help='train mode, choose between "Base" and "Classifier"')
args = parser.parse_args()

# setting up cfg
cfg._update_cfg()
cfg._print(cfg)

# torch-related setup from cfg.
device = torch.device("cuda:0" if torch.cuda.is_available() and not cfg.ignore_gpu else "cpu")
log.info(f'Using device: {device}')

cfg.seed = cfg.seed if cfg.seed else random.randint(1, 10000)
log.info('Random seed: {}'.format(cfg.seed))
torch.manual_seed(cfg.seed)
np.random.seed(cfg.seed)
random.seed(cfg.seed)

result_json = pjoin(cfg.savepath, 'result.json') if cfg.resume_result_json else None
tb_json_logger.configure(cfg.tbpath, result_json)

with open(cfg.vocab_path, 'rb')as f:
    my_vocab = dill.load(f)

# MODEL
model = RNN_WAE(n_vocab=len(my_vocab), max_seq_len=cfg.max_seq_len,
                **cfg.model).to(device)

for name_classifier in cfg.all_classifier:
    if not hasattr(model, name_classifier):
        classifier = build_classifier('dnn', cfg.model.z_dim, **cfg.classifier.C_args).to(device)
        model.add_module(name_classifier, classifier)

if cfg.loadpath:
    if os.path.exists(cfg.loadpath):
        model.load_state_dict(torch.load(cfg.loadpath), strict=False)
        log.info('Loaded model from ' + cfg.loadpath)
        print('Loaded model from ' + cfg.loadpath)
    else:
        log.info('Train new model')
        print('Train new model')

log.info(model)

# DATA
classifier_train_dataset = dataset.construct_dataset(
    cfg.dataset.train_data_path,
    cfg.dataset.current_label,
    cfg.dataset.vocab_path,
    cfg.max_seq_len)

classifier_test_dataset = dataset.construct_dataset(
    cfg.dataset.test_data_path,
    cfg.dataset.current_label,
    cfg.dataset.vocab_path,
    cfg.max_seq_len)

# train_classifier(cfg=cfg, model=model, train_dataset=classifier_train_dataset, 
#                  test_dataset=classifier_test_dataset, device=device)
model.eval()
preds_labels,auc = test_classifier(cfg=cfg, model=model,
            dataset=classifier_test_dataset, device=device)
preds = preds_labels[:,0].cpu().numpy()
labels = preds_labels[:,1].cpu().numpy()
breakpoint()