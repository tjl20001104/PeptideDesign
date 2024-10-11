import os
import torch
import dill
import numpy as np
import warnings

from data import dataset
from models.model import RNN_WAE
from models.classifier import build_classifier
from optim.test_optim import NSGAIII_searching
# from optim.GA_optim import NSGAIII_searching
import cfg
from utils import decode_from_z

warnings.filterwarnings("ignore")
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
        model.load_state_dict(torch.load(cfg.loadpath), strict=True)
        print('Loaded model from ' + cfg.loadpath)
    else:
        print('Train new model')

model.eval()

targets = [1, 0]
Boundary = [-4, 4]
Thresholds = [0.8, 0.1]
is_continue = False
F, selected_position, Population = NSGAIII_searching(model, cfg.dataset.label_name, 
            targets, Boundary, Thresholds, 100000, 1000, classifier_train_dataset)
np.save('optim/F.npy', F)
np.save('optim/selected_position.npy', selected_position)
np.save('optim/Population.npy', Population)
print('Searching Done!!!')