import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import EdgePooling, global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
import torch.nn.functional as F
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
    elif config['pooling'] == 'atomic':
        pool = global_add_pool
    else:
        raise ValueError("Invalid graph pooling type.")

    return pool

def semi_orthogonal_glorot_weights(n_in, n_out, scale=2.0, seed=None):
    W = semi_orthogonal_matrix(n_in, n_out, seed=seed)
    W *= np.sqrt(scale / ((n_in + n_out) * W.var()))
    return torch.Tensor(W).type(floating_type).t()
    
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
