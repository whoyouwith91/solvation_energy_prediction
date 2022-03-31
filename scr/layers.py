from typing import Union, Tuple, Callable
from torch_geometric.typing import OptTensor, OptPairTensor, Adj, Size
from torch.nn import Parameter

import numpy as np
import torch
import torch.nn as nn
from torch.nn import GRU, LSTM
import torch.nn.functional as F
from torch.nn import ModuleList, Sequential, Linear, ReLU, LeakyReLU, ELU, Tanh
from torch_scatter import scatter_mean
from torch_scatter import scatter
from typing import Optional, List, Dict
from torch_geometric.typing import Adj, OptTensor

from torch_geometric.nn import MessagePassing
from torch_geometric.nn import EdgePooling, global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set, TopKPooling, SAGPooling
import torch.nn.functional as F
from torch_scatter import scatter_add
from torch_geometric.nn.inits import glorot, zeros
from k_gnn import avg_pool, add_pool, max_pool
from helper import *

def dmpnn_pool(atom_hiddens, a_scope):
    mol_vecs = []
    for i, (a_start, a_size) in enumerate(a_scope):
        if a_size == 0:
            cached_zero_vector = nn.Parameter(torch.zeros(atom_hiddens.shape[-1]), requires_grad=False)
            mol_vecs.append(cached_zero_vector)
        else:
            cur_hiddens = atom_hiddens.narrow(0, a_start, a_size)
            mol_vec = cur_hiddens  # (num_atoms, hidden_size)
            mol_vec = mol_vec.sum(dim=0)
            mol_vecs.append(mol_vec)

    mol_vecs = torch.stack(mol_vecs, dim=0)  # (num_molecules, hidden_size)

    return mol_vecs  # num_molecules x hidden

def PoolingFN(config):
    #Different kind of graph pooling
    if config['gnn_type'] == 'dmpnn':
        pool = dmpnn_pool
    elif config['pooling'] == "sum":
        pool = global_add_pool
    elif config['pooling'] == "mean":
        pool = global_mean_pool
    elif config['pooling'] == "max":
        pool = global_max_pool
    elif config['pooling'] == "attention":
        if config['JK'] == "concat":
            pool = GlobalAttention(gate_nn = torch.nn.Linear((config['num_layer'] + 1) * config['emb_dim'], 1))
        else:
            pool = GlobalAttention(gate_nn = torch.nn.Linear(config['emb_dim'], 1))
    elif config['pooling'] == "set2set":
        set2set_iter = 2 # 
        if config['JK'] == "concat":
            pool = Set2Set((config['num_layer'] + 1) * config['emb_dim'], set2set_iter)
        else:
            pool = Set2Set(config['emb_dim'], set2set_iter)
    elif config['pooling'] == 'conv':
        poolList = []
        poolList.append(global_add_pool)
        poolList.append(global_mean_pool)
        poolList.append(global_max_pool)
        poolList.append(GlobalAttention(gate_nn = torch.nn.Linear(self.emb_dim, 1)))
        poolList.append(Set2Set(config['emb_dim'], 2))
        pool = nn.Conv1d(len(poolList), 1, 2, stride=2)
    elif config['pooling'] == 'edge':
        pool = []
        pool.extend([EdgePooling(config['emb_dim']).cuda() for _ in range(config['num_layer'])])
    elif config['pooling'] == 'topk':
        pool = []
        pool.extend([TopKPooling(config['dimension']).cuda() for _ in range(config['num_layer'])])
    elif config['pooling'] == 'sag':
        pool = []
        pool.extend([SAGPooling(config['dimension']).cuda() for _ in range(config['num_layer'])])
    elif config['pooling'] == 'atomic':
        pool = global_add_pool
    else:
        raise ValueError("Invalid graph pooling type.")

    return pool

class ResidualLayer(nn.Module):
    """
    The residual layer defined in PhysNet
    """
    def __init__(self, module, dim, activation, drop_ratio=0., batch_norm=False):
        super().__init__()
        self.batch_norm = batch_norm
        self.drop_ratio = drop_ratio
        self.activation = activation
        self.module = module

        self.lin1 = nn.Linear(dim, dim)
        self.lin1.weight.data = semi_orthogonal_glorot_weights(F, F)
        self.lin1.bias.data.zero_()
        if self.batch_norm:
            self.bn1 = nn.BatchNorm1d(dim, momentum=1.)

        self.lin2 = nn.Linear(dim, dim)
        self.lin2.weight.data = semi_orthogonal_glorot_weights(dim, dim)
        self.lin2.bias.data.zero_()
        if self.batch_norm:
            self.bn2 = nn.BatchNorm1d(dim, momentum=1.)

    def forward(self, module_type, x, edge_index=None, edge_attr=None):
        ### in order of Skip >> BN >> ReLU
        if module_type == 'linear': 
            x_res = self.module(x)
        elif module_type in ['gineconv', 'pnaconv', 'nnconv']:
            gnn_x = self.module(x, edge_index, edge_attr)
            x_res = gnn_x
        else:  # conv without using edge attributes
            gnn_x = self.module(x, edge_index)
            x_res = gnn_x
        
        if self.batch_norm:
            x = self.bn1(x)
        x = self.activation(x)
        x = self.lin1(x)
        x = F.dropout(x, self.drop_ratio, training = self.training)  
        if self.batch_norm:
            x = self.bn2(x)
        x = self.activation(x)
        
        x = self.lin2(x)
        x = F.dropout(x, self.drop_ratio, training = self.training)
        return x + x_res
