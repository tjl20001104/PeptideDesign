import torch
import torch.nn as nn
import torch.nn.functional as F


def build_discriminator(input_dim, **C_args):
    discriminator = CNNdiscriminator(input_dim, **C_args)
    return discriminator


class CNNdiscriminator(nn.Module):
    """
    Sequence discriminator based on a CNN architecture (Kim, 2014)
    """

    def __init__(self,
                 input_dim,
                 min_filter_width,
                 max_filter_width,
                 num_filters,
                 dropout):
        super(CNNdiscriminator, self).__init__()
        self.max_filter_width = max_filter_width

        self.conv_layers = nn.ModuleList([nn.Conv2d(1,
                                                    num_filters,
                                                    (width, input_dim))
                                          for width in range(min_filter_width, max_filter_width + 1)])

        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_filters * (max_filter_width - min_filter_width + 1), (max_filter_width - min_filter_width + 1)),
            nn.LeakyReLU(),
            nn.Linear((max_filter_width - min_filter_width + 1),1)
        )

    def forward(self, x):
        """
        Inputs must be embeddings: mbsize x seq_len x emb_dim
        """
        x = x.unsqueeze(1)  # mbsize x 1 x seq_len x emb_dim
        assert x.size(2) >= self.max_filter_width, 'Current classifier arch needs at least seqlen {}'.format(
            self.max_filter_width)

        # Compute filter outputs
        features = []
        for ix, filters in enumerate(self.conv_layers):
            cur_layer = F.relu(filters(x)).squeeze(3)
            cur_pooled = F.max_pool1d(cur_layer, cur_layer.size(2)).squeeze(2)
            features.append(cur_pooled)

        # Build feature vector
        x = torch.cat(features, dim=1)

        # Compute distribution over c in output layer
        p_c = self.fc(x)

        return x, p_c