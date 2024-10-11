import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


def build_classifier(classifier_type, emb_dim, **C_args):
    # TODO: IF/ELSE over other classifier types
    if classifier_type == 'cnn':
        classifier = CNNClassifier(emb_dim, **C_args)
    elif classifier_type == 'dnn':
        classifier = DNNClassifier(emb_dim, **C_args)
    else:
        raise ValueError('Please use CNN classifier')
    return classifier


class CNNClassifier(nn.Module):
    """
    Sequence classifier based on a CNN architecture (Kim, 2014)
    """

    def __init__(self,
                 emb_dim,
                 min_filter_width,
                 max_filter_width,
                 num_filters,
                 dropout):
        super(CNNClassifier, self).__init__()
        self.max_filter_width = max_filter_width

        self.conv_layers = nn.ModuleList([nn.Conv1d(1,
                                                    num_filters,
                                                    width)
                                          for width in range(min_filter_width, max_filter_width + 1)])

        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_filters * (max_filter_width - min_filter_width + 1), num_filters),
            nn.LeakyReLU(),
            nn.Linear(num_filters, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Inputs must be embeddings: mbsize x z_dim
        """
        x = x.unsqueeze(1)  # mbsize x 1 x z_dim
        assert x.size(2) >= self.max_filter_width, 'Current classifier arch needs at least seqlen {}'.format(
            self.max_filter_width)

        # Compute filter outputs
        features = []
        for ix, filters in enumerate(self.conv_layers):
            cur_layer = F.relu(filters(x))
            cur_pooled = F.max_pool1d(cur_layer, cur_layer.size(2)).squeeze(2)
            features.append(cur_pooled)

        # Build feature vector
        x = torch.cat(features, dim=1)

        # Compute distribution over c in output layer
        p_c = self.fc(x)
        
        # output 2 dim : [prediction for amp, prediction for tox]

        return p_c

class DNNClassifier(nn.Module):
    def __init__(self,
                 emb_dim,
                 num_layers,
                 hidden_dim,
                 dropout):
        super(DNNClassifier, self).__init__()

        fc : List[nn.Module] = []
        fc.append(nn.Linear(emb_dim,2*hidden_dim))
        fc.append(nn.BatchNorm1d(2*hidden_dim))
        fc.append(nn.LeakyReLU())
        fc.append(nn.Dropout(dropout))

        fc.extend([nn.Linear(2*hidden_dim,hidden_dim),
                   nn.LeakyReLU(),
                   nn.BatchNorm1d(hidden_dim)])

        for i in range(num_layers-3):
            fc.extend([nn.Linear(hidden_dim,hidden_dim),
                       nn.LeakyReLU(),
                       nn.BatchNorm1d(hidden_dim)])

        fc.extend([nn.Linear(hidden_dim,int(hidden_dim/4)),
                   nn.LeakyReLU(),
                   nn.BatchNorm1d(int(hidden_dim/4))])

        fc.append(nn.Linear(int(hidden_dim/4),1))
        fc.append(nn.Sigmoid())
        self.fc = nn.Sequential(*fc)

        # fc.append(nn.Linear(emb_dim,hidden_dim))
        # fc.append(nn.BatchNorm1d(hidden_dim))
        # fc.append(nn.LeakyReLU())
        # fc.append(nn.Dropout(dropout))
        # for i in range(num_layers-1):
        #     fc.extend([nn.Linear(hidden_dim,hidden_dim),
        #                nn.LeakyReLU(),
        #                nn.BatchNorm1d(hidden_dim)])
        # fc.append(nn.Linear(int(hidden_dim),1))
        # fc.append(nn.Sigmoid())
        # self.fc = nn.Sequential(*fc)

    def forward(self, x):
        """
        Inputs must be embeddings: mbsize x z_dim
        """
        p_c = self.fc(x)
        return p_c