import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F

# The GraphConvolution Layer of GRAPE
class GraphConvolution(Module):
    def __init__(self, in_features, out_features, nrole, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.norm = torch.nn.LayerNorm(out_features)
        self.weight11 = Parameter(torch.FloatTensor(in_features, out_features))
        self.weight12 = Parameter(torch.FloatTensor(out_features, out_features))
        self.filter = Parameter(torch.FloatTensor(nrole))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight11.size(1))
        self.weight11.data.uniform_(-stdv, stdv)
        stdv = 1. / math.sqrt(self.weight12.size(1))
        self.weight12.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
        stdv = 1. / math.sqrt(self.filter.size(0))
        self.filter.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):        
        aggFeature = torch.cat([torch.unsqueeze(torch.spmm(adjtemp, input), 2) for adjtemp in adj], dim=2)
        output = torch.matmul(aggFeature, F.softmax(self.filter, dim=0))
        output = torch.mm(F.relu(torch.mm(output, self.weight11)), self.weight12)
        output = self.norm(output)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
