import torch
from torch_geometric.data import (InMemoryDataset,Data)
from torch_geometric.data import Batch


#------------------Naive-------------------------------------
class GraphDataset(InMemoryDataset):
    def __init__(self,
                 root,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None):
        super(GraphDataset, self).__init__(root, transform, pre_transform, pre_filter)
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
                mol_y=d['mol_y'],
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

class GraphDataset_dmpnn_mol(InMemoryDataset):
    def __init__(self,
                 root,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None):
        super(GraphDataset_dmpnn_mol, self).__init__(root, transform, pre_transform, pre_filter)
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
                mol_y=d['mol_y'],
                N=d['N'],
                Z=d['Z'],
                a2b=d['a2b'],
                b2a=d['b2a'],
                b2revb=d['b2revb'],
                n_atoms=d['n_atoms'],
                n_bonds=d['n_bonds']
                ) for d in raw_data_list
        ]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
#--------------------------------------------DMPNN-----------------------------------------------


#--------------------------------------------SingleTask----------------------------------------------
class GraphDataset_single(InMemoryDataset):
    def __init__(self,
                 root,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None):
        super(GraphDataset_single, self).__init__(root, transform, pre_transform, pre_filter)
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
                mol_y=d['mol_y'],
                N=d['N'],
                Z=d['Z']
                ) for d in raw_data_list
        ]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
#---------------------------------------------SingleTask---------------------------------------------

def collate_dmpnn(data_list):
    """
    A :class:`BatchMolGraph` represents the graph structure and featurization of a batch of molecules.

    A BatchMolGraph contains the attributes of a :class:`MolGraph` plus:

    * :code:`atom_fdim`: The dimensionality of the atom feature vector.
    * :code:`bond_fdim`: The dimensionality of the bond feature vector (technically the combined atom/bond features).
    * :code:`a_scope`: A list of tuples indicating the start and end atom indices for each molecule.
    * :code:`b_scope`: A list of tuples indicating the start and end bond indices for each molecule.
    * :code:`max_num_bonds`: The maximum number of bonds neighboring an atom in this batch.
    * :code:`b2b`: (Optional) A mapping from a bond index to incoming bond indices.
    * :code:`a2a`: (Optional): A mapping from an atom index to neighboring atom indices.
    """
    if 'mol_y' in data_list[0]:
        keys = ['N', 'Z',  'mol_y']
    
    atom_fdim = data_list[0]['x'].shape[1]
    bond_fdim = atom_fdim+7
    batch = Batch()
    #batch.batch = []
    
    
    for key in keys:      
        batch[key] = []
     # Start n_atoms and n_bonds at 1 b/c zero padding
    n_atoms = 1  # number of atoms (start at 1 b/c need index 0 as padding)
    n_bonds = 1  # number of bonds (start at 1 b/c need index 0 as padding)
    a_scope = []  # list of tuples indicating (start_atom_index, num_atoms) for each molecule
    b_scope = []  # list of tuples indicating (start_bond_index, num_bonds) for each molecule

    # All start with zero padding so that indexing with zero padding returns zeros
    f_atoms = [[0] * atom_fdim]  # atom features
    f_bonds = [[0] * bond_fdim]  # combined atom/bond features
    a2b = [[]]  # mapping from atom index to incoming bond indices
    b2a = [0]  # mapping from bond index to the index of the atom the bond is coming from
    b2revb = [0]  # mapping from bond index to the index of the reverse bond
    #batch['Z'] = [0]
    for i, mol_graph in enumerate(data_list): 
        batch['mol_y'].append(mol_graph.mol_y)
        batch['N'].append(mol_graph.N)
        batch['Z'].append(mol_graph.Z)
        
        f_atoms.extend(mol_graph.x)
        f_bonds.extend(mol_graph.edge_attr)
        
        for a in range(mol_graph.n_atoms):
            a2b.append([b + n_bonds for b in mol_graph.a2b[a]])

        for b in range(mol_graph.n_bonds):
            b2a.append(n_atoms + mol_graph.b2a[b])
            b2revb.append(n_bonds + mol_graph.b2revb[b])

        a_scope.append((n_atoms, mol_graph.n_atoms.item()))
        b_scope.append((n_bonds, mol_graph.n_bonds.item()))
        n_atoms += mol_graph.n_atoms.item()
        #print(a_scope)
        n_bonds += mol_graph.n_bonds.item()

    max_num_bonds = max(1, max(
            len(in_bonds) for in_bonds in a2b))  # max with 1 to fix a crash in rare case of all single-heavy-atom mols

    f_atoms = torch.FloatTensor(f_atoms)
    f_bonds = torch.FloatTensor(f_bonds)
    a2b = torch.LongTensor([a2b[a] + [0] * (max_num_bonds - len(a2b[a])) for a in range(n_atoms)])
    b2a = torch.LongTensor(b2a)
    b2revb = torch.LongTensor(b2revb)
    
    for key in keys:
        #print(key)
        if torch.is_tensor(batch[key][0]):
            batch[key] = torch.cat(
                batch[key], dim=data_list[0].__cat_dim__(key, batch[key][0]))
    
    batch.x = f_atoms
    batch.edge_attr = f_bonds
    batch.a2b = a2b
    batch.b2a = b2a
    batch.b2revb = b2revb
    batch.a_scope = a_scope
    batch.b_scope = b_scope
    
    return batch.contiguous()

class DataLoader_dmpnn(torch.utils.data.DataLoader):
    def __init__(self, dataset, **kwargs):
        super(DataLoader_dmpnn, self).__init__(dataset, collate_fn=collate_dmpnn, **kwargs)