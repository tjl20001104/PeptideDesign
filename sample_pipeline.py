import torch

import os
import sys
import pprint
import argparse
from collections import OrderedDict
import datetime
import json
import h5py

from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
from modlamp.analysis import GlobalAnalysis
from modlamp.descriptors import GlobalDescriptor

import cfg

from data import dataset
from evals import PeptideEvaluator
from density_modeling import mogQ, evaluate_nll

from api import (load_trained_model,
                 Vocab,
                 get_model_and_vocab_path)

import logging

LOG = logging.getLogger('GenerationAPI')
logging.basicConfig(
    format='%(asctime)s %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p',
    level=logging.INFO)
pp = pprint.PrettyPrinter(indent=2, depth=1)

Q_CLASS = mogQ
Q_KWARGS = {'n_components': None,
            'z_num_samples': 10,
            'covariance_type': None}


def get_encodings_from_dataloader(model, dataset, num_batch, device):
    LOG.info('Start encoding {} samples from dataset'.format(num_batch))
    mus, logvars, labels = [], [], []
    loader = enumerate(dataset)
    for batch in range(num_batch):
        with torch.no_grad():
            _,item = next(loader)
            inputs = item[0].to(device)
            inputs = inputs.unsqueeze(0)
            (mu, logvar), z, (dec_logits,dec_logits_gau) = model(
                inputs, sample_z='max')
            mus.append(mu.detach())
            logvars.append(logvar.detach())
            labels.append(item[1].unsqueeze(0))
    mus, logvars, labels = torch.cat(mus, dim=0), torch.cat(logvars, dim=0), torch.cat(labels, dim=0)
    return mus, logvars, labels


def fitQ_and_test(QClass, QKwargs, Q_select={}, negative_select={}, model=None, dataset=None, device=None):
    """
    Fit Q_xi^a(z) based on samples with single
    "attribute=y" <- Q_select query.
    Collect metrics: Q_xi^a(z),
                     p(z)
                     nll on heldout positive and
                     heldout negative samples.
    """
    mu, logvar, _ = get_encodings_from_dataloader(model=model, dataset=dataset, num_batch=1000, device=device)

    Q_xi_a = QClass(mu, logvar, **QKwargs)
    
    LOG.info('Fitted {}  {} on selection {}'.format(QClass.__name__,
                                                    str(QKwargs),
                                                    str(Q_select)))

    eval_points = [
        ('a,tr', (mu, logvar))
    ]

    metrics = OrderedDict()
    for name, points in eval_points:
        nllq, nllp = evaluate_nll(Q_xi_a, points)
        # key = r'CE$(q^{{ {} }} |{{}})$'.format(name)
        metrics[name] = (nllq, nllp)
    return Q_xi_a, metrics


def decode_from_z(z, model, dataset):
    sall = []
    LOG.info('Decoder decoding: beam search')
    for zchunk in torch.split(z, 1000):
        s, _ = model.generate_sentences(zchunk.size(0),
                                           zchunk,
                                           sample_mode='beam',
                                           beam_size=5)
        s = [hypotheses[0] for hypotheses in s]
        sall += s
    return dataset.idx2sentences(sall, print_special_tokens=False)


def save_csv_pkl(samples, fn):
    outfn = fn + '.csv'
    samples.drop(columns='z').to_csv(outfn, index_label='idx')
    outfn = fn + '.pkl'
    samples.to_pickle(outfn)


def save_samples(samples, basedir, fn_prefix):
    outfn = os.path.join(basedir, fn_prefix)
    outfn += '_{}'.format(datetime.datetime.now().isoformat().split('T')[0])
    with open(outfn + '.plain.txt', 'w') as fh:
        fh.write(samples['peptide'].to_string(index=False))
    save_csv_pkl(samples, outfn)
    LOG.info('Full sample list written to {}.pkl/csv'.format(outfn))
    accepted = samples[samples.accept]
    accepted_fn = '{}.accepted.{}'.format(outfn, len(accepted))
    save_csv_pkl(accepted, accepted_fn)
    LOG.info('Accepted sample list written to {}.pkl/csv'.format(accepted_fn))


# def score_lclfZ(clf, z):
#     z = z.numpy()
#     RETURN_LABEL_COL_IX = 1
#     probs = clf.predict_proba(z)[:, RETURN_LABEL_COL_IX]
#     return probs


# def score_clfZ(clf, z):
#     z = z.numpy()
#     RETURN_LABEL_COL_IX = 1
#     probs = clf.predict_proba(z)[:, RETURN_LABEL_COL_IX]
#     return probs


# def build_lclfZ(attr, model, dataset, device):
#     """
#     sklearn logistic reg clf between attr=1 and attr=0.
#     based on vis/scripts/tsne.py

#     ASSUMES that for attr we get -1, 0, 1 labels,
#     corresponding to na / neg / pos
#     """
#     z_mu, z_logvar, z_labels = get_encodings_from_dataloader(model=model, dataset=dataset,
#     num_batch=1000, device=device)
#     Y = []
#     num_pos, num_neg = 0, 0
#     for label in z_labels:
#         if label[0]==1:
#             Y.append(1)
#             num_pos += 1
#         elif label[1]==1:
#             Y.append(0)
#             num_neg += 1
#         else:
#             Y.append(-1)
#     Y = torch.tensor(Y)
#     X = z_mu
#     X, Y = X.numpy(), Y.numpy()

#     clf = LogisticRegression(solver='lbfgs', max_iter=200)
#     clf.fit(X, Y)
#     acc = clf.score(X, Y)
#     LOG.info('Fitted LogReg classifier in z-space, on attr={}.'.format(
#         attr))
#     LOG.info('num samples: {} all, {} pos, {} neg. train accuracy={:.5f}'.format(
#         z_mu.shape[0], num_pos, num_neg, acc))
#     breakpoint()
#     return clf


def get_new_samples(model, dataset, Q, n_samples):
    """
    Get one round of sampled z's and decode
    """

    samples_z, scores_z, accept_z = Q.rejection_sample(n_samples=n_samples)
    samples = decode_from_z(samples_z, model, dataset)
    df = pd.DataFrame({'peptide': samples,
                       # 'sample_source': get_sample_source_str(),
                       'z': [tuple(z.tolist()) for z in samples_z],
                       'accept_z': accept_z,
                       **scores_z})
    return df


def compute_modlamp(df):
    '''
    H: hydrophobicity, the higher, the better.
    uH: hydrophobic moments,
    charge: 
    instability_index(不稳定指数): lower than 40,
    aliphatic_index(脂肪族氨基酸指数): 
    hydrophobic_ratio(疏水性): 
    '''
    ana_obj = GlobalAnalysis(df.peptide.str.replace(' ', ''))
    global_des = GlobalDescriptor(df.peptide.str.replace(' ', '').tolist())
    ana_obj.calc_H()
    ana_obj.calc_uH()
    ana_obj.calc_charge()
    global_des.instability_index()
    df.loc[:, 'instability_index'] = global_des.descriptor
    global_des.aliphatic_index()
    df.loc[:, 'aliphatic_index'] = global_des.descriptor
    global_des.hydrophobic_ratio()
    df.loc[:, 'hydrophobic_ratio'] = global_des.descriptor
    global_des.isoelectric_point()
    df.loc[:, 'isoelectric_point'] = global_des.descriptor
    df.loc[:, 'H'] = ana_obj.H[0]
    df.loc[:, 'uH'] = ana_obj.uH[0]
    df.loc[:, 'charge'] = ana_obj.charge[0]
    return df


def one_sampling_round(model, dataset, Q, n_samples_per_round):
    """
    Generate one round of samples
    """
    samples_df = get_new_samples(model, dataset, Q, n_samples_per_round)
    samples_df = compute_modlamp(samples_df)
    mask_accept = samples_df['accept_z']
    samples_df['accept'] = mask_accept
    return samples_df


def get_sample_source_str():
    return ' '.join(sys.argv[1:])


def main(args={}):
    device = torch.device("cpu")
    MODEL_PATH, VOCAB_PATH, _ = get_model_and_vocab_path()
    LOG.info('Load model, vocab, dataloader.')
    vocab = Vocab(VOCAB_PATH)
    model = load_trained_model(MODEL_PATH,
                               vocab.size()).to(device)
    LOG.info('Loaded model succesfully.')
    LOG.info('Set up dataset, evaluator objects')

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    wae_dataset = dataset.construct_dataset(
    cfg.dataset.all_data_path,
    cfg.dataset.label_name,
    cfg.dataset.vocab_path,
    cfg.max_seq_len)

    pep_evaluator = PeptideEvaluator(
        seq_len=cfg.max_seq_len)

    LOG.info('Fit attribute-conditioned marginal posterior Q_xi^a(z)')
    for k in Q_KWARGS:
        if 'Q_' + k in dir(args):
            Q_KWARGS[k] = getattr(args, 'Q_' + k)

    if args.Q_select_amppos:
        Q_SELECT_QUERY = {'amp': 1}
        Q_NEGATIVE_QUERY = {'amp': 0}
    else:
        Q_SELECT_QUERY = {}
        Q_NEGATIVE_QUERY = {}

    Q, Q_xi_metrics = fitQ_and_test(Q_CLASS,
                                    Q_KWARGS,
                                    Q_SELECT_QUERY,
                                    Q_NEGATIVE_QUERY,
                                    model,
                                    wae_dataset,
                                    device=device)
    LOG.info('Q Fit metrics: ')
    print(json.dumps(Q_xi_metrics, indent=4))

    z_clfs = {}
    for attr in ['amp', 'tox']:
        # clf_dataset = dataset.construct_dataset(
        # cfg.datapath+'/'+attr+'_lab.csv',
        # attr,
        # cfg.dataset.vocab_path,
        # cfg.max_seq_len)
        # clf_zspace = build_lclfZ(attr, model, clf_dataset, device=device)
        z_clfs[attr] = eval('model.'+'forward_'+attr+'_classifier')

    Q.init_attr_classifiers(z_clfs, clf_targets={'amp': 1, 'tox': 0})

    '''
    SETUP DONE, SAMPLING BELOW
    '''

    samples = pd.DataFrame(columns=['peptide'])
    round_ix = 0

    def is_finished(df, min_accepted):
        unfinished = len(df) < min_accepted or df['accept'].sum() < min_accepted
        return not unfinished

    while not is_finished(samples, args.n_samples_acc):
        round_ix += 1
        LOG.info("Round #{}".format(round_ix))
        new_samples = one_sampling_round(
            model,
            wae_dataset,
            Q,
            args.n_samples_per_round)

        new_samples = new_samples.loc[new_samples.peptide.drop_duplicates().index]
        new_samples = new_samples[~new_samples['peptide'].isin(
            samples['peptide'])]
        samples = pd.concat([samples, new_samples], ignore_index=True, sort=False)
        dropped_num = args.n_samples_per_round - new_samples.shape[0]
        if dropped_num > 0:
            LOG.info("Dropped {} duplicate samples".format(dropped_num))
        LOG.info('Q_xi(z|a) rejection sampling acceptance rate: {}/{} = {:.4f}'.format(
            samples['accept_z'].sum(), len(samples), 100.0 * samples['accept_z'].sum() / len(samples)))
        LOG.info('     - full filter pipeline accepted: {}/{} = {:.4f}'.format(
            samples['accept'].sum(), len(samples), 100.0 * samples['accept'].sum() / len(samples)))

    save_samples(samples, cfg.savepath, args.samples_outfn_prefix)


if __name__ == "__main__":
    LOG.info("Sample pipeline. Fit Q_xi(z), Sample from it, score samples.")
    parser = argparse.ArgumentParser(
        argument_default=argparse.SUPPRESS,
        description='Override config float & string values')
    cfg._cfg_import_export(parser, cfg, mode='fill_parser')
    parser.add_argument('--QClass', default='mogQ')
    parser.add_argument(
        '--Q_n_components', type=int, default=100,
        help='mog num components for Q model')
    parser.add_argument(
        '--Q_covariance_type', default='diag',
        help='mog Q covariance type full|tied|diag')
    parser.add_argument(
        '--n_samples_per_round', type=int, default=5000,
        help='number of samples to generate & evaluate.')
    parser.add_argument(
        '--n_samples_acc', type=int, default=100,
        help='number of samples to generate & evaluate.')
    parser.add_argument(
        '--samples_outfn_prefix', default='samples',
        help='''prefix to fn to write out the samples.
                Will have .txt .csv .pkl versions''')
    parser.add_argument(
        '--Q_select_amppos', type=int, default=0,
        help='select amp positive to fit Q_xi or not.')
    parser.add_argument(
        '--Q_from_full_dataloader', action='store_true', default=False,
        help='to fit Q_z, select from full dataloader')
    args = parser.parse_args()

    cfg._override_config(args, cfg)
    cfg._update_cfg()
    cfg._print(cfg)
    main(args)
