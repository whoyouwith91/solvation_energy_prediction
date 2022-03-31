import os, sys, math, json, argparse, logging, time, random, pickle
from datetime import datetime
import torch
import numpy as np
import pandas as pd
from featurization import *
from sklearn.model_selection import train_test_split
from mordred import Calculator, descriptors
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.rdmolfiles import SDMolSupplier, MolFromPDBFile
from rdkit.ML.Descriptors import MoleculeDescriptors
#from three_level_frag import cleavage, AtomListToSubMol, standize, mol2frag, WordNotFoundError, counter
from deepchem.feat import BPSymmetryFunctionInput, CoulombMatrix, CoulombMatrixEig
from dscribe.descriptors import ACSF
from rdkit.Chem.rdmolfiles import MolToXYZFile
from ase.io import read as ase_read
from rdkit import Chem
from prody import *
import mdtraj as md
import itertools, operator

def parse_input_arguments():
    parser = argparse.ArgumentParser(description='Physicochemical prediction')
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--file', type=str, default=None) # for file name such as pubchem, zinc, etc in Frag20
    parser.add_argument('--removeHs', action='store_true') # whether remove Hs
    parser.add_argument('--ACSF', action='store_true')
    parser.add_argument('--cutoff', type=float)
    parser.add_argument('--dmpnn', action='store_true')
    parser.add_argument('--xyz', type=str)
    parser.add_argument('--save_path', type=str, default='/scratch/dz1061/gcn/chemGraph/data/')
    parser.add_argument('--train_type', type=str)

    return parser.parse_args()

def getMol(file, id_, config):
    if config['xyz'] == 'MMFFXYZ':
        data = 'Frag20'
        format_ = '.sdf' 
    elif config['xyz'] == 'QMXYZ':
        data = 'Frag20_QM'
        format_ = '.opt.sdf' # optimized by DFT
    else:
        pass

    if file in ['pubchem', 'zinc']:
        path_to_sdf = '/ext3/{}/lessthan10/sdf/'.format(data) + file # path to the singularity file overlay-50G-10M.ext3
        sdf_file = os.path.join(path_to_sdf, str(id_)+format_)
    elif file in ['CCDC']:
        path_to_sdf = '/ext3/{}/{}/sdf'.format(data, file) # path to the singularity file overlay-50G-10M.ext3
        if config['xyz'] == 'MMFFXYZ':
            sdf_file = os.path.join(path_to_sdf, str(id_)+'_min.sdf')
        else:
            sdf_file = os.path.join(path_to_sdf, str(id_)+'.opt.sdf')
    else:
        path_to_sdf = '/ext3/{}/{}/sdf'.format(data, file)
        sdf_file = os.path.join(path_to_sdf, str(id_)+format_)
    #print(sdf_file)
    suppl = SDMolSupplier(sdf_file, removeHs=config['removeHs'])
    return suppl[0]


def main():
    args = parse_input_arguments()
    this_dic = vars(args)
    alldatapath = '/scratch/dz1061/gcn/chemGraph/data/' # TODO

    train_raw = pd.read_csv(os.path.join(alldatapath, args.dataset, 'split', 'train.csv'))
    valid_raw = pd.read_csv(os.path.join(alldatapath, args.dataset, 'split', 'valid.csv'))
    test_raw = pd.read_csv(os.path.join(alldatapath, args.dataset, 'split', 'test.csv'))
    
    examples = []
    all_data = pd.concat([train_raw, valid_raw, test_raw]).reset_index(drop=True)
        
    if this_dic['dataset'] == 'sol_calc':
        if this_dic['ACSF']:
            acsf = ACSF(
                        species=['B', 'Br', 'C', 'Cl', 'F', 'H', 'N', 'O', 'P', 'S'],
                        rcut=args.cutoff,
                        g2_params=[[1, 1], [1, 2], [1, 3]],
                        g4_params=[[1, 1, 1], [1, 2, 1], [1, 1, -1], [1, 2, -1]],)

        for value, id_, file in zip(all_data['CalcSol'], all_data['ID'], all_data['SourceFile']):
            molgraphs = {}
            mol = getMol(file, int(id_), this_dic)

            mol_graph = MolGraph(mol) # 2d or 3d
            if not this_dic['ACSF']: # 2D featurization
                molgraphs['x'] = torch.FloatTensor(mol_graph.f_atoms) # 2d

            if this_dic['ACSF']: # 3D featurization
                folder = 'Frag20' if args.xyz == 'MMFFXYZ' else 'Frag20_QM'
                if file in ['pubchem', 'zinc']:
                    path_to_xyz = '/ext3/{}/lessthan10/xyz'.format(folder) # path to the singularity file overlay-50G-10M.ext3
                else:
                    path_to_xyz = '/ext3/{}/{}/xyz'.format(folder, file)
                    file_id = str(file) +'_' + str(int(id_)) # such as 'pubchem_100001'
                    atoms = ase_read(os.path.join(path_to_xyz, '{}.xyz'.format(file_id))) # path to the singularity file overlay-50G-10M.ext3
                    molgraphs['x'] = torch.FloatTensor(acsf.create(atoms, positions=range(mol.GetNumAtoms())))
                    assert mol.GetNumAtoms() == molgraphs['x'].shape[0]
                
            if args.dmpnn: # saving features of molecular graphs for D-MPNN
                mol_graph = MolGraph_dmpnn(mol, args.ACSF, molgraphs['x'].tolist())
                molgraphs['n_atoms'] = mol_graph.n_atoms
                molgraphs['n_bonds'] = mol_graph.n_bonds
                molgraphs['a2b'] = mol_graph.a2b
                molgraphs['b2a'] = mol_graph.b2a
                molgraphs['b2revb'] = mol_graph.b2revb
                   
            atomic_number = []
            for atom in mol.GetAtoms():
                atomic_number.append(atom.GetAtomicNum())
            z = torch.tensor(atomic_number, dtype=torch.long)
            molgraphs['Z'] = z
            molgraphs['N'] = torch.FloatTensor([mol.GetNumAtoms()])

            molgraphs['edge_attr'] = torch.FloatTensor(mol_graph.f_bonds)
            molgraphs['edge_index'] = torch.LongTensor(np.concatenate([mol_graph.at_begin, mol_graph.at_end]).reshape(2,-1))
            molgraphs['mol_y'] = torch.FloatTensor([value])      
                
            examples.append(molgraphs)

            if not os.path.exists(os.path.join(this_dic['save_path'], args.dataset, 'raw')):
                os.makedirs(os.path.join(this_dic['save_path'], args.dataset, 'raw'))
            torch.save(examples, os.path.join(this_dic['save_path'], args.dataset, 'raw', 'temp.pt')) ###
            print('Finishing processing {} compounds'.format(len(examples)))

    if this_dic['dataset'] == 'freesolv':
            if this_dic['ACSF']:
                inchi_idx = pickle.load(open(os.path.join(alldatapath, '{}/split/inchi_index.pt'.format(args.dataset)), 'rb'))
                species = ['Br', 'C', 'Cl', 'F', 'H', 'I', 'N', 'O', 'P', 'S'] # no B 
                if args.train_type in ['FT']:
                    species = ['B', 'Br', 'C', 'Cl', 'F', 'H', 'N', 'O', 'P', 'S'] # no I
                periodic = False
                        

                acsf = ACSF(
                            species=species,
                            rcut=args.cutoff,
                            g2_params=[[1, 1], [1, 2], [1, 3]],
                            g4_params=[[1, 1, 1], [1, 2, 1], [1, 1, -1], [1, 2, -1]],
                            periodic=periodic)

            for inchi, tar in zip(all_data['InChI'], all_data['target']):
                molgraphs = {}
                        
                idx = inchi_idx[inchi]
                mol = SDMolSupplier(os.path.join(alldatapath, args.dataset, 'split', 'sdf', '{}.sdf'.format(idx)), removeHs=args.removeHs)[0]
                if args.train_type in ['FT', 'TL'] and not \
                        set([atom.GetSymbol() for atom in mol.GetAtoms()]) < set(['B', 'Br', 'C', 'Cl', 'F', 'H', 'N', 'O', 'P', 'S']):
                            continue
                        
                mol_graph = MolGraph(mol)
                
                if not this_dic['ACSF']:
                    molgraphs['x'] = torch.FloatTensor(mol_graph.f_atoms)
                else:      
                    if not os.path.exists(os.path.join(alldatapath, args.dataset, 'split', this_dic['xyz'], '{}.xyz'.format(idx))):
                        continue
                    atoms = ase_read(os.path.join(alldatapath, args.dataset, 'split', this_dic['xyz'], '{}.xyz'.format(idx)))
                    molgraphs['x'] = torch.FloatTensor(acsf.create(atoms, positions=range(mol.GetNumAtoms())))

                if args.dmpnn:
                    mol_graph = MolGraph_dmpnn(mol, args.ACSF, molgraphs['x'].tolist())
                    molgraphs['n_atoms'] = mol_graph.n_atoms
                    molgraphs['n_bonds'] = mol_graph.n_bonds
                    molgraphs['a2b'] = mol_graph.a2b
                    molgraphs['b2a'] = mol_graph.b2a
                    molgraphs['b2revb'] = mol_graph.b2revb

                atomic_number = []
                for atom in mol.GetAtoms():
                    atomic_number.append(atom.GetAtomicNum())
                z = torch.tensor(atomic_number, dtype=torch.long)
                molgraphs['Z'] = z

                molgraphs['N'] = torch.FloatTensor([mol.GetNumAtoms()])                    
                molgraphs['edge_attr'] = torch.FloatTensor(mol_graph.f_bonds)
                molgraphs['edge_index'] = torch.LongTensor(np.concatenate([mol_graph.at_begin, mol_graph.at_end]).reshape(2,-1))
                molgraphs['mol_y'] = torch.FloatTensor([tar])          
                molgraphs['InChI'] = inchi
                molgraphs['id'] = idx

                examples.append(molgraphs)
                
                if not os.path.exists(os.path.join(this_dic['save_path'], args.dataset, 'raw')):
                    os.makedirs(os.path.join(this_dic['save_path'], args.dataset, 'raw'))
                torch.save(examples, os.path.join(this_dic['save_path'], args.dataset, 'raw', 'temp.pt')) ###
                print('Finishing processing {} compounds'.format(len(examples)))
        

if __name__ == '__main__':
    main()

