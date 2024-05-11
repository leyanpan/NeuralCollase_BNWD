import torch
from torch import nn
from torch_utils import dist_from_etf

class MLP(nn.Module):
    def __init__(self, layer_widths, bn=True, weight_norm=False, inst_norm=False, layer_norm=False, activation='ReLU', linear_bias=False, bn_eps=1e-5, bn_affine=True):
        if activation == 'ReLU':
          act_layer = nn.ReLU
        if activation == 'tanh':
          act_layer = nn.Tanh
        super().__init__()
        self.layer_widths = layer_widths
        self.weight_norm = weight_norm
        self.inst_norm = inst_norm
        self.layer_norm = layer_norm
        self.bn = bn
        layers = []
        for i in range(len(layer_widths) - 2):
          layers.append(nn.Linear(layer_widths[i], layer_widths[i+1], bias=linear_bias))
          layers.append(act_layer())
        if self.bn:
          layers.append(torch.nn.BatchNorm1d(layer_widths[i+1], eps=bn_eps, affine=bn_affine))
        self.last_layer = nn.Linear(layer_widths[-2], layer_widths[-1], bias=linear_bias)
        self.feature = nn.Sequential(*layers)


    def forward(self, x):
        x = x.view(x.shape[0], -1)
        features = self.feature(x)
        return self.last_layer(features)

    def last_layer_feat(self, x):
        return self.feature(x)

    def all_features(self, x):
        features = []
        for i in range(len(self.feature)):
          x = self.feature[i](x)
          if isinstance(self.feature[i], nn.Linear):
            features.append(x)
        x = self.last_layer(x)
        features.append(x)
        return features

    def nc_loss(self, x, y):
        loss = torch.tensor(0)
        decay_fac = 0.75
        all_feats = self.all_features(x)
        cur_fac = decay_fac ** len(all_feats)
        for feature in all_feats:
          loss = loss + cur_fac * dist_from_etf(feature, y)
          cur_fac /= decay_fac
        return loss

    def num_layers(self):
        return len(self.model)

    def normalize_weight(self):
      with torch.no_grad():
        for name, p in self.named_parameters():
          if 'weight' in name or 'bias' in name:
            p /= torch.norm(p)


    def layer_feat(self, x, i):
        for j in range(i):
            x = self.model[j](x)
        return x

    def last_layer_weight(self):
        return self.model[-1].weight