import os
import torch
import numpy as np
from functools import reduce
import operator
import torch.nn as nn


def describe(t):  # t could be numpy or torch tensor.
    t = t.data if isinstance(t, torch.autograd.Variable) else t
    s = '{:17s} {:8s} [{:.4f} , {:.4f}] m+-s = {:.4f} +- {:.4f}'
    ttype = 'np.{}'.format(t.dtype) if type(t) == np.ndarray else str(t.type()).replace('ensor', '')
    si = 'x'.join(map(str, t.shape if isinstance(t, np.ndarray) else t.size()))
    return s.format(ttype, si, t.min(), t.max(), t.mean(), t.std())


def write_gen_samples(samples, fn, c_lab=None):
    """ samples: list of strings. c_lab (optional): tensor of same size. """
    fn_dir = os.path.dirname(fn)
    if not os.path.exists(fn_dir):
        os.makedirs(fn_dir)

    size = len(samples)
    with open(fn, 'w+') as f:
        if c_lab is not None:
            print("Saving %d samples with labels" % size)
            assert c_lab.nelement() == size, 'sizes dont match'
            f.writelines(['label: {}\n{}\n'.format(y, s) for y, s in zip(c_lab, samples)])
        else:
            print("Saving %d samples without labels" % size)
            f.write('\n'.join(samples) + '\n')


def accuracy_pred(preds_labels,threshold=0.5):
    preds = preds_labels[:,0]
    labels = preds_labels[:,1]
    preds = torch.where(preds>threshold, torch.ones_like(preds), torch.zeros_like(preds))
    return torch.eq(preds, labels).float().mean()


def clf_pred(labels, df, model, dataset):
    model.eval()
    sequences = df['peptide'].tolist()
    sequences = [line.split(' ') for line in sequences]
    len_list = torch.tensor([len(line)+2 for line in sequences]).to(model.device)
    sequences = torch.stack(dataset.sentences2idx(sequences)).to(model.device)
    (z_mu, z_logvar), z, (dec_logits_z, dec_logits_gau) = model(sequences, len_list, sample_z='max')
    for label in labels:
        classifier = getattr(model, label+'_classifier')
        classifier.eval()
        preds = classifier(z)
        df[label] = preds.detach().cpu().numpy()
    return df


def decode_from_z(z, model, dataset):
    model.eval()
    sall = []
    for zchunk in torch.split(z, 1000):
        s, _ = model.generate_sentences(zchunk.size(0),
                zchunk,
                sample_mode='categorical')
        sall += s
    return dataset.idx2sentences(sall, print_special_tokens=False)


# Linearly interpolate between start and end val depending on current iteration
def interpolate(start_val, end_val, start_iter, end_iter, current_iter):
    if current_iter < start_iter:
        return start_val
    elif current_iter >= end_iter:
        return end_val
    else:
        return start_val + (end_val - start_val) * (current_iter - start_iter) / (end_iter - start_iter)


def anneal(cfgan, it):
    return interpolate(cfgan.start.val, cfgan.end.val, cfgan.start.iter, cfgan.end.iter, it)


def check_dir_exists(fn):
    fn_dir = os.path.dirname(fn)
    if not os.path.exists(fn_dir):
        os.makedirs(fn_dir)


def prod(iterable):
    return reduce(operator.mul, iterable, 1)


def scale_and_clamp(dist, w, clamp_val=None):
    rescaled = dist * w  # w = 1/scale
    if clamp_val and rescaled > clamp_val:
        return clamp_val
    else:
        return rescaled

def free_params(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = True

def frozen_params(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = False

def check_is_nan(tensor):
    if torch.isnan(tensor):
        return torch.tensor(0.0,requires_grad=True)
    else:
        return tensor