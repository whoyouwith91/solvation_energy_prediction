from dataProcess import *
from k_gnn import DataLoader
from DataPrepareUtils import my_pre_transform #sPhysnet
from utils_functions import collate_fn # sPhysnet
import torch
import random

        
class get_data_loader():
    def __init__(self, config):
        self.config = config
        self.model = config['model']
      
        self.train_size, self.val_size = config['train_size'], config['val_size']
        if config['sample'] or config['vary_train_only']:
            self.data_seed = config['data_seed']
        self.train_loader, self.val_loader, self.test_loader, self.num_features, self.num_bond_features = self.graph_loader()
      
    
    def graph_loader(self):
        if self.config['dataset'] == 'sol_calc':
            if self.config['propertyLevel'] == 'molecule': # naive, only with solvation property
                if self.config['gnn_type'] == 'dmpnn':
                    dataset = GraphDataset_dmpnn_mol(root=self.config['data_path'])
                else:
                    dataset = GraphDataset_single(root=self.config['data_path'])
         
        else: # for typical other datasets 
            if self.config['gnn_type'] == 'dmpnn':
                dataset = GraphDataset_dmpnn_mol(root=self.config['data_path'])
            else:
                dataset = GraphDataset(root=self.config['data_path'])
      
        num_features = dataset.num_features
        num_bond_features = dataset[0]['edge_attr'].shape[1]
          
        my_split_ratio = [self.train_size, self.val_size]

        if self.config['sample']:
            random.seed(self.data_seed)
            if self.config['fix_test']:
                rest_dataset = dataset[:my_split_ratio[0]+my_split_ratio[1]]
                train_dataset = rest_dataset.index_select(random.sample(range(my_split_ratio[0]+my_split_ratio[1]), my_split_ratio[0]))
                val_dataset = rest_dataset[list(set(rest_dataset.indices()) - set(train_dataset.indices()))]
                test_dataset = dataset[my_split_ratio[0]+my_split_ratio[1]:]
            
            elif self.config['vary_train_only']: # varying train size and random split for train/valid/test
                rest_dataset = dataset.index_select(random.sample(range(len(dataset)), my_split_ratio[0]+my_split_ratio[1]))
                test_dataset = dataset[list(set(dataset.indices()) - set(rest_dataset.indices()))]
                train_dataset = rest_dataset[:my_split_ratio[0]]
                val_dataset = rest_dataset[my_split_ratio[0]:my_split_ratio[0]+my_split_ratio[1]]
                train_dataset = train_dataset.index_select(random.sample(range(my_split_ratio[0]), self.config['sample_size']))
            else: # random split for train/valid/test
                rest_dataset = dataset.index_select(random.sample(range(len(dataset)), my_split_ratio[0]+my_split_ratio[1]))
                test_dataset = dataset[list(set(dataset.indices()) - set(rest_dataset.indices()))]
                train_dataset = rest_dataset[:my_split_ratio[0]]
                val_dataset = rest_dataset[my_split_ratio[0]:my_split_ratio[0]+my_split_ratio[1]]
      
        else: # not sampling
            if not self.config['vary_train_only']:
                test_dataset = dataset[my_split_ratio[0]+my_split_ratio[1]:]
                rest_dataset = dataset[:my_split_ratio[0]+my_split_ratio[1]]
                train_dataset, val_dataset = rest_dataset[:my_split_ratio[0]], rest_dataset[my_split_ratio[0]:]
            else:
                random.seed(self.data_seed)
                test_dataset = dataset[my_split_ratio[0]+my_split_ratio[1]:]
                rest_dataset = dataset[:my_split_ratio[0]+my_split_ratio[1]]
                train_dataset, val_dataset = rest_dataset[:my_split_ratio[0]], rest_dataset[my_split_ratio[0]:]
                train_dataset = train_dataset.index_select(random.sample(range(my_split_ratio[0]), self.config['sample_size']))

        if self.config['gnn_type'] == 'dmpnn':
            test_loader = DataLoader_dmpnn(test_dataset, batch_size=self.config['batch_size'], num_workers=0)
            val_loader = DataLoader_dmpnn(val_dataset, batch_size=self.config['batch_size'], num_workers=0)
            train_loader = DataLoader_dmpnn(train_dataset, batch_size=self.config['batch_size'], num_workers=0, shuffle=True)
        else:
            test_loader = DataLoader(test_dataset, batch_size=self.config['batch_size'], num_workers=0)
            val_loader = DataLoader(val_dataset, batch_size=self.config['batch_size'], num_workers=0)
            train_loader = DataLoader(train_dataset, batch_size=self.config['batch_size'], num_workers=0, shuffle=True)
      
      return train_loader, val_loader, test_loader, num_features, num_bond_features




