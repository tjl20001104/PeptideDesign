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
import lightgbm as lgb
from lightgbm import LGBMClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score as AUC
from sklearn.metrics import roc_curve, accuracy_score, precision_recall_curve

# amp_df_train = pd.read_csv('zs_amp_train.csv',index_col=0).values
# amp_df_test = pd.read_csv('zs_amp_test.csv',index_col=0).values
# tox_df_train = pd.read_csv('zs_tox_train.csv',index_col=0).values
# tox_df_test = pd.read_csv('zs_tox_test.csv',index_col=0).values

# clf = SVC(C=1,kernel="rbf")
# gbm = LGBMClassifier()

# x_train = df_train[:,:100]
# y_train = df_train[:,-1]
# train_dataset = lgb.Dataset(x_train, y_train)

# x_test =df_test[:,:100]
# y_test = df_test[:,-1]
# test_dataset = lgb.Dataset(x_test, y_test)

# params = {
#     'boosting_type': 'gbdt',
#     'objective': 'binary',
#     'metric': {'auc'},
#     'num_leaves' : 16,
#     'learning_rate' : 0.01,
#     'lambda_l1' : 10,
#     'lambda_l2' : 10,
#     'bagging_freq' : 5
# }

# gbm = lgb.train(params, train_dataset, num_boost_round=100)
# # gbm.fit(x_train, y_train, num_leaves=64, n_estimators=300, learning_rate=0.01, reg_alpha=10, reg_lambda=10)

# # train_score = accuracy_score(y_train, gbm.predict(x_train))
# # test_score = accuracy_score(y_test, gbm.predict(x_test))

# auc_score_train = AUC(y_train, gbm.predict(x_train), labels=[0,1])
# auc_score_test = AUC(y_test, gbm.predict(x_test), labels=[0,1])
# # print(train_score,test_score)
# print(auc_score_train,auc_score_test)

# amp_preds_train = amp_df_train[:,100]
# amp_labels_train = amp_df_train[:,101]>0.5
# amp_preds_test = amp_df_test[:,100]
# amp_labels_test = amp_df_test[:,101]>0.5

# tox_preds_train = tox_df_train[:,100]
# tox_labels_train = tox_df_train[:,101]>0.5
# tox_preds_test = tox_df_test[:,100]
# tox_labels_test = tox_df_test[:,101]>0.5

# preds = preds_train
# labels = labels_train

# preds = preds_test
# labels = labels_test

# precisions, recalls, thresholds = precision_recall_curve(labels,preds)
# f1_scores = (2 * precisions * recalls) / (precisions + recalls)
# best_f1_score = np.max(f1_scores[np.isfinite(f1_scores)])
# best_f1_score_index = np.argmax(f1_scores[np.isfinite(f1_scores)])
# best_thres = thresholds[best_f1_score_index]



# fpr, tpr, thresholds = roc_curve(amp_labels_test,amp_preds_test,pos_label=1)
# amp_auc = AUC(amp_labels_test, amp_preds_test)
# tox_auc = AUC(tox_labels_test, tox_preds_test)
# breakpoint()

# print(best_f1_score, best_thres)
# print(auc)
# plt.plot(fpr,tpr)
# plt.show()

cfg._update_cfg()

# torch-related setup from cfg.
device = torch.device("cuda:0" if torch.cuda.is_available() and not cfg.ignore_gpu else "cpu")

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
        print('Loaded model from ' + cfg.loadpath)
    else:
        print('Train new model')

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

model.eval()
preds_labels,auc = test_classifier(cfg=cfg, model=model,
            dataset=classifier_test_dataset, device=device)
preds = preds_labels[:,0].cpu().numpy()
labels = preds_labels[:,1].cpu().numpy()
breakpoint()
# fpr, tpr, thresholds = roc_curve(labels,preds,pos_label=1)
# plt.plot(fpr,tpr)
# plt.title('auc curve of classifier on {}'.format(cfg.dataset.current_label))
# plt.legend(labels=['auc={:.2f}'.format(auc)],loc='best', fontsize=10, ncol=1,
#            labelcolor='c')
# plt.show()