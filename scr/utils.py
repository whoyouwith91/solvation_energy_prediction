from featurization import *
from torch_geometric.data import (InMemoryDataset,Data)
from rdkit.Chem.rdmolfiles import SDMolSupplier
from dscribe.descriptors import ACSF
from ase.io import read as ase_read
import torch
import numpy as np
import os, json

def generate_graphs(sdf_file, xyz_file):
    molgraphs = {}
    
    species = ['B', 'Br', 'C', 'Cl', 'F', 'H', 'N', 'O', 'P', 'S']
    acsf = ACSF(
                species=species,
                rcut=6.0,
                g2_params=[[1, 1], [1, 2], [1, 3]],
                g4_params=[[1, 1, 1], [1, 2, 1], [1, 1, -1], [1, 2, -1]],
                periodic=False)
    mol = SDMolSupplier(sdf_file, removeHs=False)[0]
    mol_graph = MolGraph(mol)
    
    atoms = ase_read(xyz_file)
    molgraphs['x'] = torch.FloatTensor(acsf.create(atoms, positions=range(mol.GetNumAtoms())))
    
    atomic_number = []
    for atom in mol.GetAtoms():
        atomic_number.append(atom.GetAtomicNum())
    z = torch.tensor(atomic_number, dtype=torch.long)
    molgraphs['Z'] = z

    molgraphs['N'] = torch.FloatTensor([mol.GetNumAtoms()])                    
    molgraphs['edge_attr'] = torch.FloatTensor(mol_graph.f_bonds)
    molgraphs['edge_index'] = torch.LongTensor(np.concatenate([mol_graph.at_begin, mol_graph.at_end]).reshape(2,-1))

    return molgraphs

class GraphDataset_test(InMemoryDataset):
    def __init__(self,
                 root,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None):
        super(GraphDataset_test, self).__init__(root, transform, pre_transform, pre_filter)
        self.type = type
        self.data, self.slices = torch.load(self.processed_paths[0])
    
    @property
    def raw_file_names(self):
        return 'temp.pt'

    @property
    def processed_file_names(self):
        return 'processed.pt'
    
    def download(self):
        pass

    def process(self):
        raw_data_list = torch.load(self.raw_paths[0])
        data_list = [
            Data(
                x=d['x'],
                edge_index=d['edge_index'],
                edge_attr=d['edge_attr'],
                Z=d['Z'],
                N=d['N'],
                ) for d in raw_data_list
        ]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
#------------------Naive-------------------------------------

def loadConfig(path, name='config.json'):
    with open(os.path.join(path,name), 'r') as f:
        config = json.load(f)
    return config