import math

import numpy as np
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

def parse_tree(tree):
    def parse_node(node):
        i = node.find(':')
        if i == -1:
            return node, ''
        return node[i+1:], node[:i]
    node_map = dict()
    parent = dict()
    pstack = [0]
    counter = 0
    node = ""
    # ret_exp = ""
    for char in tree:
        if char in ('(', ')'):
            if node:
                node_name, node_type = parse_node(node)
                node_map[counter] = {'name': node_name, 'type': node_type, 'idx': counter}
                # ret_exp += str(counter)
                parent[counter] = pstack[-1]
                pstack.append(counter)
                counter += 1
            if char == ')':
                pstack.pop()
            # ret_exp += char
            node = ""
        elif char == ' ':
            continue
        else:
            node += char
    return node_map, parent

def process_tree(tree, wdim):
    tree_map, parent = parse_tree(tree)
    # print(tree_map)
    # print(parent)
    N = len(tree_map)
    x = np.random.rand(N, wdim)
    adj = np.identity(N)
    for jj, i in parent.items():
        adj[i][tree_map[jj]['idx']] = 1.0
    return x, adj

class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class EdGCN(Module):
    def __init__(self, wdim, dropout=0.5):
        super(EdGCN, self).__init__()
        self.gc1 = GraphConvolution(wdim, wdim)
        self.gc2 = GraphConvolution(wdim, wdim)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return x
