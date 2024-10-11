import sys, os, types
import json
from collections import OrderedDict
from utils import check_dir_exists


# small helper stuff
class Bunch(dict):
    def __init__(self, *args, **kwds):
        super(Bunch, self).__init__(*args, **kwds)
        self.__dict__ = self

def _print(cfg_, prefix=''):
    for k in dir(cfg_):
        if k[0] == '_': continue  # hidden
        v = getattr(cfg_, k)
        if type(v) in [float, str, int, bool]:
            print('{}{}\t{}'.format(prefix, k, v))
        elif type(v) == Bunch:  # recurse; descend into Bunch
            print('{}{}:'.format(prefix, k))
            _print(v, prefix + '  |- ')

def _override_config(args, cfg):
    """ call _cfg_import_export in override mode, update cfg from:
        (1) contents of config_json (taken from (a) loadpath if not auto, or (2) savepath)
        (2) from command line args
    """
    config_json = vars(args).get('config_json', '')
    _cfg_import_export(args, cfg, mode='override')


def _override_config_from_json(cfg, config_json):
    if config_json:
        override_vals = Bunch(json.load(open(config_json)))
    # Now actually import into cfg
    _cfg_import_export(override_vals, cfg, mode='override')


def _save_config(cfg_overrides, cfg_complete, savepath):
    json_fn = os.path.join(savepath, 'config_overrides.json')
    check_dir_exists(json_fn)
    with open(json_fn, 'w') as fh:
        json.dump(vars(cfg_overrides), fh, indent=2, sort_keys=True)
    json_fn = os.path.join(savepath, 'config_complete.json')
    with open(json_fn, 'w') as fh:
        d = {}
        _cfg_import_export(d, cfg_complete, mode='fill_dict')
        json.dump(d, fh, indent=2, sort_keys=True)
    # add if desired: _copy_to_nested_dict(cfg_complete) dump


def _copy_to_nested_dict(cfg_):
    """ follows _cfg_import_export() flow but creates nested dictionary """
    ret = {}
    for k in dir(cfg_):
        if k[0] == '_': continue  # hidden
        v = getattr(cfg_, k)
        if type(v) in [float, str, int, bool]:
            ret[k] = v
        elif type(v) == Bunch:  # recurse; descend into Bunch
            ret[k] = _copy_to_nested_dict(v)
    return ret


def _cfg_import_export(cfg_interactor, cfg_, prefix='', mode='fill_parser'):
    """ Iterate through cfg_ module/object. For known variables import/export
    from cfg_interactor (dict, argparser, or argparse namespace) """
    for k in dir(cfg_):
        if k[0] == '_': continue  # hidden
        v = getattr(cfg_, k)
        if type(v) in [float, str, int, bool]:
            if mode == 'fill_parser':
                cfg_interactor.add_argument('--{}{}'.format(prefix, k), type=type(v), help='default: {}'.format(v))
            elif mode == 'fill_dict':
                cfg_interactor['{}{}'.format(prefix, k)] = v
            elif mode == 'override':
                prek = '{}{}'.format(prefix, k)
                if prek in cfg_interactor:
                    setattr(cfg_, k, getattr(cfg_interactor, prek))
        elif type(v) == Bunch:  # recurse; descend into Bunch
            _cfg_import_export(cfg_interactor, v, prefix=prefix + k + '.', mode=mode)

def _update_cfg():
    """ function to update/postprocess based on special cfg values """
    global wae , classifier, dataset, runname, seed, savepath_toplevel, savepath, loadpath, datapath, \
        resume_result_json, tb_toplevel, tbpath, all_classifier, vocab_path
    # dataset, dataset_unl, dataset_lab
    # constructing savepath and resultpath
    savepath = os.path.join(savepath_toplevel, runname)  # {savepath}/model_{iter}.pt
    tbpath = os.path.join(tb_toplevel, runname)  # {tbpath}/eventfiles

    # inject shared fields into wae
    wae.update(shared)
    classifier.update(shared)

    # Vocab path
    dataset.vocab_path = os.path.join(datapath,'vocab')
    dataset.all_data_path = os.path.join(datapath,'all_amp.csv')

    # for classifier
    labels = ['amp','tox']
    all_classifier = [label + '_classifier' for label in labels]
    dataset.label_name = labels
    dataset.current_label = 'tox'
    dataset.current_classifier = dataset.current_label + '_classifier'
    dataset.train_data_path = os.path.join(datapath,dataset.current_label+'_train.csv')
    dataset.test_data_path = os.path.join(datapath,dataset.current_label+'_test.csv')

    # checkpoint paths: inject into cfgv, and use to define auto-loadpath.
    chkpt_path = os.path.join(savepath, 'model_{}.pt')
    wae.chkpt_path = chkpt_path
    classifier.chkpt_path = chkpt_path

    # Load path
    # loadpath = chkpt_path.format(wae.s_iter)
    loadpath = chkpt_path.format(classifier.s_iter)
    
    # seeding
    # if seed:  # increment the seed to have new seeds per sub-run: different loader shuffling, model/training stochasticity
    #     seed += (phase - 1) * partN + part

    # set result fns
    def set_result_filenames(cfgv, savepath, list_of_fns):
        for fieldname, fn in list_of_fns:
            cfgv[fieldname] = os.path.join(savepath, fn)

    set_result_filenames(wae, savepath,
                         [('gen_samples_path', 'wae_gen.txt'), ('eval_path', 'wae_eval.txt'),
                          ('fasta_gen_samples_path', 'wae_gen.fasta')])

# general
ignore_gpu = False
# seed = 1238
seed = None
max_seq_len = 50
all_classifier = ''

# paths
savepath_toplevel = 'output'  # output/run_name_with_hypers/{checkpoints, generated sequences, etc}
loadpath = ''  # autofill: savepath + right iter based in s_iter
datapath = 'data'
# runname = 'classifier_1'
runname = 'final'
resume_result_json = True  # load up and append to result.json by default
tb_toplevel = 'tb'
vocab_path = os.path.join(datapath,'vocab')

# wae
wae = Bunch(
    batch_size=32,
    lr_G=3e-5,
    lr_D=1e-6,
    n_critic=5,
    # TODO lrate decay with scheduler
    s_iter=1700,
    n_iter=1500,
    beta=Bunch(
        start=Bunch(val=1.0, iter=0),
        end=Bunch(val=1.0, iter=1000)
    ),
    lambda_logvar_L1=0.0,  # default from https://openreview.net/pdf?id=r157GIJvz
    lambda_logvar_Dis=1e-3,  # default from https://openreview.net/pdf?id=r157GIJvz
    lambda_recon=0.1,
    cheaplog_every=1,  # cheap tensorboard logging eg training metrics
    expsvlog_every=100,  # expensive logging: model checkpoint, heldout set evals, word emb logging
)
wae.beta.start.iter = wae.s_iter
wae.beta.end.iter = wae.s_iter + wae.n_iter


# classifier
classifier = Bunch(
    batch_size=32,
    lr_C=1e-7,  # classifier,
    # TODO lrate decay with scheduler
    s_iter=200,
    n_iter=200,
    lambda_logvar_L2=1e-2,
    classifier_min_length=5,  # specific to classifier architecture
    # hypers for controlled text gen
    beta=Bunch(
        start=Bunch(val=1.0, iter=wae.n_iter),
        end=Bunch(val=1.0, iter=wae.n_iter + 4000)
    ),
    cheaplog_every=1,  # cheap tensorboard logging eg training metrics
    expsvlog_every=5,  # expensive logging: model checkpoint, heldout set evals, word emb logging
    # C_args=Bunch(
    #     min_filter_width=3,
    #     max_filter_width=15,
    #     num_filters=20,
    #     dropout=0.1
    # ),
    C_args=Bunch(
        hidden_dim=200,
        num_layers=4,
        dropout=0.1
    )
)
classifier.beta.start.iter = classifier.s_iter
classifier.beta.end.iter = classifier.n_iter


# shared settings, are injected in train & full Bunch in _update_cfg()
shared = Bunch(
    clip_value=5.0,
    clip_grad=5.0
)

# evals settings
evals = Bunch(
    sample_size=2000,
    sample_modes=Bunch(
        # cat  = Bunch(sample_mode='categorical', temp=0.8),
        beam=Bunch(sample_mode='beam', beam_size=5, n_best=3),
    ),
)

losses = Bunch(
    wae_mmd=Bunch(
        sigma=7.0,  # ~ O( sqrt(z_dim) )
        kernel='gaussian',
        # for method = rf
        rf_dim=500,
        rf_resample=False
    ),
)

# model architecture
model = Bunch(
    z_dim=100,
    emb_dim=150,
    pretrained_emb=None,  # set True to load from dataset_unl.get_vocab_vectors()
    freeze_embeddings=False,
    E_args=Bunch(
        h_dim=80,  # 20 for amp, 64 for yelp
        biGRU=True,
        layers=1,
        p_dropout=0.0
    ),
    G_args=Bunch(
        G_class='gru',
        GRU_args=Bunch(
            # h_dim = (z_dim + c_dim) for now. TODO parametrize this?
            p_word_dropout=0.3,
            p_out_dropout=0.3,
            skip_connetions=False,
        ),
        deconv_args=Bunch(
            max_seq_len=max_seq_len,
            num_filters=100,
            kernel_size=4,
            num_deconv_layers=3,
            useRNN=False,
            temperature=1.0,
            use_batch_norm=True,
            num_conv_layers=2,
            add_final_conv_layer=True,
        ),
    ),
    D_args=Bunch(
        min_filter_width=3,
        max_filter_width=5,
        num_filters=100,
        dropout=0.5
    ),
    C_args=Bunch(
        min_filter_width=3,
        max_filter_width=5,
        num_filters=100,
        dropout=0.5
    )
)

# dataset
dataset = Bunch(
    train_data_path='',
    test_data_path='',
    all_data_path='',
    vocab_path='',
    label_name=None,
    current_label=None,
    current_classifier=None,
)