import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution
import torch
from torch.nn.parameter import Parameter
import math

# GRAPE Model
class GRAPE(nn.Module):
    def __init__(self, nlayer, nfeat, nhid, nclass, nrole, ngene, dropout, attn):
        super(GRAPE, self).__init__()
        if nlayer == 1:
            self.gc = [nn.ModuleList([GraphConvolution(nfeat, nclass, nrole[ind]) for ind in range(ngene)])]
        else:
            self.gc = [nn.ModuleList([GraphConvolution(nfeat, nhid, nrole[ind]) for ind in range(ngene)])]
            for ind in range(nlayer-2):
                self.gc.append(nn.ModuleList([GraphConvolution(nhid, nhid, nrole[ind]) for ind in range(ngene)]))
            self.gc.append(nn.ModuleList([GraphConvolution(nhid, nclass, nrole[ind]) for ind in range(ngene)]))
        self.gc = nn.ModuleList(self.gc)

        self.dropout = dropout
        self.ngene = ngene
        self.attn = attn
        self.nlayer = nlayer
        if self.attn:
            self.Wk = Parameter(torch.FloatTensor(nlayer, ngene, ngene))
            self.Wq = Parameter(torch.FloatTensor(nlayer, ngene, ngene))
            self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.ngene)
        self.Wk.data.uniform_(-stdv, stdv)
        self.Wq.data.uniform_(-stdv, stdv)

    def forward(self, x, adj):
        # squeeze and excitation
        for layer_ind in range(self.nlayer):
            x = F.dropout(x, self.dropout, training=self.training)
            x = torch.cat([torch.unsqueeze(F.relu(self.gc[layer_ind][ind](x, adj[ind])), 1) for ind in range(self.ngene)], 1)
            if self.attn:
                sq_x = torch.mean(x, dim=[0, 2])
                k = F.relu(torch.matmul(sq_x, self.Wk[layer_ind]))
                self.att_weight = torch.unsqueeze(torch.unsqueeze(F.softmax(torch.matmul(k, self.Wq[layer_ind]), dim=0), 0), 2)
                x = torch.mul(self.att_weight, x)
                x = torch.sum(x, 1)
            else:
                x = torch.mean(x, 1)
        return x
