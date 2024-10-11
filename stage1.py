import logging
import os
from os.path import join as pjoin
import torch
import numpy as np
import random
import argparse

from data import dataset
from models.model import RNN_WAE
from train_wae import train_wae
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


# DATA
wae_dataset = dataset.construct_dataset(
    cfg.dataset.all_data_path,
    cfg.dataset.label_name,
    cfg.dataset.vocab_path,
    cfg.max_seq_len)

# MODEL
model = RNN_WAE(n_vocab=wae_dataset.n_vocab, max_seq_len=cfg.max_seq_len,
                **cfg.model).to(device)
log.info(model)

if cfg.loadpath:
    if os.path.exists(cfg.loadpath):
        model.load_state_dict(torch.load(cfg.loadpath))
        log.info('Loaded model from ' + cfg.loadpath)
    else:
        log.info('Train new model')


train_wae(cfg=cfg, model=model, dataset=wae_dataset, device=device)
