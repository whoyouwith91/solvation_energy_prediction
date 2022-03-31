import torch
from torch import nn
from torch_geometric.nn.glob.glob import global_add_pool
from torch_scatter import scatter_mean
from torch_geometric.utils import add_self_loops, degree, softmax
import torch.nn.functional as F
from torch_scatter import scatter_add
from torch_geometric.nn.inits import glorot, zeros
from helper import *
from layers import *
from gnns import *
from PhysDimeNet import PhysDimeNet
from torch_geometric.nn.norm import PairNorm
import time, sys

def get_model(config):
    name = config['model']
    if name == None:
        raise ValueError('Please specify one model you want to work on!')
    if name == '1-GNN':
        return GNN_1(config)
    if name == 'physnet':
        return PhysDimeNet(**config)

class GNN(torch.nn.Module):
    """
    Basic GNN unit modoule.
    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        JK (str): last, concat, max or sum.
        max_pool_layer (int): the layer from which we use max pool rather than add pool for neighbor aggregation
        drop_ratio (float): dropout rate
        gnn_type: gin, gcn, graphsage, gat
    Output:
        node representations
    """
    def __init__(self, config):
        super(GNN, self).__init__()
        self.config = config
        self.num_layer = config['num_layer']
        self.emb_dim = config['emb_dim']
        self.drop_ratio = config['drop_ratio']
        self.gnn_type = config['gnn_type']
        self.bn = config['bn']
        self.act_fn = activation_func(config)

        if self.num_layer < 1:
            raise ValueError("Number of GNN layers must be no less than 1.")
        if self.config['gnn_type'] in ['gineconv', 'pnaconv', 'nnconv']:
            self.linear_b = Linear(self.config['num_bond_features'], self.emb_dim)
        
        # define graph conv layers
        if self.gnn_type in ['dmpnn']:
            self.gnns = get_gnn(self.config) # already contains multiple layers
        else:
            self.linear_x = Linear(self.config['num_atom_features'], self.emb_dim)
            self.gnns = nn.ModuleList()
            for _ in range(self.num_layer):
                self.gnns.append(get_gnn(self.config).model())
        
        ###List of batchnorms
        if self.bn and self.gnn_type not in ['dmpnn']:
            self.batch_norms = nn.ModuleList()
            for _ in range(self.num_layer):
                self.batch_norms.append(nn.BatchNorm1d(self.emb_dim))

    def forward(self, *argv):
        if len(argv) == 3:
            x, edge_index, edge_attr = argv[0], argv[1], argv[2]
        elif len(argv) == 4:
            x, edge_index, edge_attr, batch = argv[0], argv[1], argv[2], argv[3]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        elif len(argv) == 5:
            f_atoms, f_bonds, a2b, b2a, b2revb = argv[0], argv[1], argv[2], argv[3], argv[4]
        else:
            raise ValueError("unmatched number of arguments.")

        if self.gnn_type == 'dmpnn':
            node_representation = self.gnns(f_atoms, f_bonds, a2b, b2a, b2revb)
        else:
            x = self.linear_x(x) # first linear on atoms 
            h_list = [x]
            #x = F.relu(self.linear_x(x))
            if self.config['gnn_type'] in ['gineconv', 'pnaconv', 'nnconv']:
                edge_attr = self.linear_b(edge_attr.float()) # first linear on bonds 
                #edge_attr = F.relu(self.linear_b(edge_attr.float())) # first linear on bonds 

            for layer in range(self.num_layer):
                if self.config['residual_connect']: # adding residual connection
                    residual = h_list[layer] 
                if self.config['gnn_type'] in ['gineconv', 'pnaconv', 'nnconv']:
                    h = self.gnns[layer](h_list[layer], edge_index, edge_attr)
                elif self.config['gnn_type'] in ['dnn']:
                    h = self.gnns[layer](h_list[layer])
                else:
                    h = self.gnns[layer](h_list[layer], edge_index)
                
                ### in order of Skip >> BN >> ReLU
                if self.config['residual_connect']:
                    h += residual
                if self.bn:
                    h = self.batch_norms[layer](h)
                
                #h = self.pair_norm(h, data.batch)
                if layer == self.num_layer - 1:
                    #remove relu for the last layer
                    h = F.dropout(h, self.drop_ratio, training = self.training)
                    self.last_conv = self.get_activations(h)
                else:
                    h = self.act_fn(h)
                    h = F.dropout(h, self.drop_ratio, training = self.training)
                h_list.append(h)
            
            node_representation = h_list[-1]
        
        return node_representation

class GNN_1(torch.nn.Module):
    def __init__(self, config):
        super(GNN_1, self).__init__()
        self.config = config
        self.dataset = config['dataset']
        self.num_layer = config['num_layer']
        self.fully_connected_layer_sizes = config['fully_connected_layer_sizes']
        self.emb_dim = config['emb_dim']
        self.graph_pooling = config['pooling']
        self.propertyLevel = config['propertyLevel']
        self.gnn_type = config['gnn_type']

        self.gnn = GNN(config)
        self.outLayers = nn.ModuleList()

        #For graph-level property predictions
        if self.graph_pooling == "set2set": # set2set will double dimension
            self.mult = 2
        else:
            self.mult = 1
        embed_size = self.mult * self.emb_dim 

        for idx, (L_in, L_out) in enumerate(zip([embed_size] + self.fully_connected_layer_sizes, self.fully_connected_layer_sizes + [self.num_tasks])):
            if idx != len(self.fully_connected_layer_sizes):
                fc = nn.Sequential(Linear(L_in, L_out), activation_func(config), nn.Dropout(config['drop_ratio']))
                self.outLayers.append(fc)
            else:
                fc = nn.Sequential(Linear(L_in, L_out), nn.Dropout(config['drop_ratio']))
                self.outLayers.append(fc)
                    
        if self.config['normalize']:
            shift_matrix = torch.zeros(self.emb_dim, 1)
            scale_matrix = torch.zeros(self.emb_dim, 1).fill_(1.0)
            shift_matrix[:, :] = self.config['energy_shift'].view(1, -1)
            scale_matrix[:, :] = self.config['energy_scale'].view(1, -1)
            self.register_parameter('scale', torch.nn.Parameter(scale_matrix, requires_grad=True))
            self.register_parameter('shift', torch.nn.Parameter(shift_matrix, requires_grad=True))

    def forward(self, data):
        if self.gnn_type == 'dmpnn':
             f_atoms, f_bonds, a2b, b2a, b2revb, a_scope = data.x, data.edge_attr, data.a2b, data.b2a, data.b2revb, data.a_scope
        else:
            x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr.long(), data.batch
        
        if self.config['normalize']:
            Z = data.Z
            if self.gnn_type == 'dmpnn':
               Z = torch.cat((torch.zeros(1).int(), data.Z))
        
        if self.gnn_type == 'dmpnn':
            node_representation = self.gnn(f_atoms, f_bonds, a2b, b2a, b2revb) # node updating 
        elif self.gnn_type == 'dnn':
            node_representation = self.gnn(data) # node updating 
        else:
            node_representation = self.gnn(x, edge_index, edge_attr) # node updating
        
        # graph pooling 
        if self.graph_pooling == 'atomic': # 
            MolEmbed = node_representation #(-1, emb_dim)
        else: # normal pooling functions besides conv and edge
            MolEmbed = self.pool(node_representation, batch)  # atomic read-out (-1, 1)
        
        for layer in self.outLayers: # 
                MolEmbed = layer(MolEmbed)
            if self.config['normalize']:
                MolEmbed = self.scale[Z, :] * MolEmbed + self.shift[Z, :]
                
        if self.graph_pooling == 'atomic':
            if self.gnn_type == 'dmpnn':
                return self.pool(MolEmbed, a_scope).view(-1)
            else: # pnaconv, ginconv
                return global_add_pool(MolEmbed, batch).view(-1)
            

