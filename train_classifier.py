import sys
from tqdm import tqdm

import torch
import torch.optim as optim
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score as AUC

from models.mutils import save_model
import utils
from models import losses
from torch.utils.data import DataLoader
from tb_json_logger import log_value


def train_classifier(cfg, model, train_dataset, test_dataset, device):
    cfgv = cfg.classifier
    dataloader = DataLoader(train_dataset, batch_size=cfgv.batch_size, shuffle=True)
    print('Training {} classifier ...'.format(cfg.dataset.current_label))

    classifier = getattr(model, cfg.dataset.current_classifier)
    trainer_Classifier = optim.RMSprop(classifier.parameters(), lr=cfgv.lr_C, weight_decay=cfgv.lambda_logvar_L2)

    for it in tqdm(range(cfgv.s_iter, cfgv.s_iter + cfgv.n_iter + 1), disable=None):
        if it % cfgv.cheaplog_every == 0 or it % cfgv.expsvlog_every == 0:
            def tblog(k, v):
                log_value('train_' + k, v, it)
        else:
            tblog = lambda k, v: None

        utils.frozen_params(model)
        utils.free_params(classifier)
        model.train()

        for index,item in enumerate(dataloader):
            inputs = item[0].to(device)
            labels = item[1].to(device)
            input_lens = item[2].to(device)
            trainer_Classifier.zero_grad()

            # ============ Train Classifier ============ #

            (z_mu, z_logvar), z, (dec_logits_z, dec_logits_gau) = model(inputs, input_lens, sample_z='max')
            preds = classifier(z)
            loss_classifier = losses.loss_classifier(preds, labels)

            loss_classifier.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(classifier.parameters(), cfgv.clip_grad)
            trainer_Classifier.step()

            tblog('loss_classifier', loss_classifier.item())

        if it % cfgv.cheaplog_every == 0 or it % cfgv.expsvlog_every == 0:
            model.eval()
            preds_labels, auc_score = test_classifier(cfg, model, train_dataset, device)
            accuracy = utils.accuracy_pred(preds_labels)
            tblog('accuracy_train', accuracy.item())
            tblog('Grad_norm', grad_norm)
            tblog('auc_train', auc_score)
            # tqdm.write(
            #     'ITER {} TRAINING (phase 1). loss_classifier: {:.4f};'
            #     'Grad_norm: {:.4e} '
            #         .format(it, loss_classifier.item(), grad_norm))
            # sys.stdout.flush()
            if it % cfgv.expsvlog_every == 0 and it > cfgv.s_iter:
                preds_labels, auc_score = test_classifier(cfg, model, test_dataset, device)
                accuracy = utils.accuracy_pred(preds_labels)
                tblog('auc_test', auc_score)
                tblog('accuracy_test', accuracy.item())
                save_model(model, cfgv.chkpt_path.format(it))

def test_classifier(cfg, model, dataset, device):
    cfgv = cfg.classifier
    dataloader = DataLoader(dataset, batch_size=cfgv.batch_size, shuffle=True)

    classifier = getattr(model, cfg.dataset.current_classifier)
    preds_labels = torch.Tensor().to(device)
    zs = torch.Tensor().to(device)

    utils.frozen_params(model)
    
    for index,item in enumerate(dataloader):
        inputs = item[0].to(device)
        labels = item[1].to(device)
        input_lens = item[2].to(device)

        (z_mu, z_logvar), z, (dec_logits_z, dec_logits_gau) = model(inputs, input_lens, sample_z='max')
        preds = classifier(z)
        preds_labels = torch.cat([preds_labels, torch.cat([preds, labels],dim=1)])
        zs = torch.cat([zs,z],dim=0)
    
    # preds_labels = preds_labels.cpu()
    # zs = zs.cpu()
    # array = torch.cat([zs,preds_labels],dim=1)
    # df = pd.DataFrame(array)
    # # df.to_csv('zs_{}_train.csv'.format(cfg.dataset.current_label))
    # df.to_csv('zs_{}_test.csv'.format(cfg.dataset.current_label))

    preds_labels_np = preds_labels.cpu().numpy()
    preds = preds_labels_np[:,0]
    labels = preds_labels_np[:,1]
    auc_score = AUC(labels, preds)

    return preds_labels, auc_score