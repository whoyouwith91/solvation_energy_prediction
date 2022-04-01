import os, sys
import torch
import numpy as np
import pandas as pd
from featurization import *
from rdkit.Chem.rdmolfiles import SDMolSupplier
from dscribe.descriptors import ACSF
from ase.io import read as ase_read
from rdkit import Chem

def parse_input_arguments():
    parser = argparse.ArgumentParser(description='Solvation energy dataset preparation')
    parser.add_argument('--data_path', type=str) # where csv files are saved.
    parser.add_argument('--dataset', type=str) # Frag20-Aqsol-100K or FreeSolv
    parser.add_argument('--ACSF', action='store_true') # using 3D features or not 
    parser.add_argument('--cutoff', type=float) # cutoff values in ACSF
    parser.add_argument('--dmpnn', action='store_true') # using D-MPNN-way to do featurization
    parser.add_argument('--xyz', type=str) # using MMFF or QM optimized geometries
    parser.add_argument('--train_type', type=str) # training from scratch (TS) or finetuning (TS)

    return parser.parse_args()

def getMol(file_, id_, config):
    if config['xyz'] == 'MMFF':
        format_ = '.sdf' 
    elif config['xyz'] == 'QM':
        format_ = '.opt.sdf' # optimized by DFT
    else:
        pass  

    if file_ in ['pubchem', 'zinc']:
        path_to_sdf = './data/Frag20-Aqsol-100K/sdf/{}/lessthan10/{}'.format(config['xyz'], file_) # path to the singularity file overlay-50G-10M.ext3
        sdf_file = os.path.join(path_to_sdf, str(id_)+format_)
    elif file_ in ['CCDC']:
        path_to_sdf = './data/Frag20-Aqsol-100K/sdf/{}/{}'.format(config['xyz'], file_) # path to the singularity file overlay-50G-10M.ext3
        if config['xyz'] == 'MMFF':
            sdf_file = os.path.join(path_to_sdf, str(id_)+'_min.sdf')
        else:
            sdf_file = os.path.join(path_to_sdf, str(id_)+'.opt.sdf')
    else:
        path_to_sdf = './data/Frag20-Aqsol-100K/sdf/{}/{}/'.format(config['xyz'], file_)
        sdf_file = os.path.join(path_to_sdf, str(id_)+format_)
    #print(sdf_file)
    suppl = SDMolSupplier(sdf_file, removeHs=False)
    return suppl[0]


def main():
    args = parse_input_arguments()
    this_dic = vars(args)
    
    ###### read in the raw csv datasets. 
    train_raw = pd.read_csv(os.path.join(args.data_path, args.dataset, 'split', 'train.csv'))
    valid_raw = pd.read_csv(os.path.join(args.data_path, args.dataset, 'split', 'valid.csv'))
    test_raw = pd.read_csv(os.path.join(args.data_path, args.dataset, 'split', 'test.csv'))
    all_data = pd.concat([train_raw, valid_raw, test_raw]).reset_index(drop=True)

    examples = []
    if this_dic['dataset'] == 'sol_calc': # For Frag20-Aqsol-100K
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
                if file_ in ['pubchem', 'zinc']:
                    path_to_xyz = './data/Frag20-Aqsol-100K/xyz/{}/lessthan10/{}'.format(args.xyz, file_) # path to the singularity file overlay-50G-10M.ext3
                else:
                    path_to_xyz = './data/Frag20-Aqsol-100K/xyz/{}/{}'.format(args.xyz, file_)
                file_id = str(file_) +'_' + str(int(id_)) # such as 'pubchem_100001'
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
            
            style = '3D' if this_dic['ACSF'] else '2D'
            if not os.path.exists(os.path.join(this_dic['save_path'], args.dataset, style, args.xyz, 'graphs/raw')):
                os.makedirs(os.path.join(this_dic['save_path'], args.dataset, style, args.xyz, 'graphs/raw'))
            torch.save(examples, os.path.join(this_dic['save_path'], args.dataset, style, args.xyz, 'graphs/raw', 'temp.pt')) ###
            print('Finishing processing {} compounds'.format(len(examples)))

    if this_dic['dataset'] == 'freesolv': # For FreeSolv
            if this_dic['ACSF']:
                inchi_idx = pickle.load(open(os.path.join(args.data_path, 'FreeSolv/split/inchi_index.pt'), 'rb')) # InChI mapped to index pointing to the SDF and XYZ files. 
                species = ['Br', 'C', 'Cl', 'F', 'H', 'I', 'N', 'O', 'P', 'S'] 
                if args.train_type in ['FT']:
                    species = ['B', 'Br', 'C', 'Cl', 'F', 'H', 'N', 'O', 'P', 'S'] 
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
                mol = SDMolSupplier(os.path.join(args.data_path, 'FreeSolv/sdf/{}.sdf'.format(idx)), removeHs=False)[0]
                if args.train_type in ['FT', 'TL'] and not \
                        set([atom.GetSymbol() for atom in mol.GetAtoms()]) < set(['B', 'Br', 'C', 'Cl', 'F', 'H', 'N', 'O', 'P', 'S']):
                            continue
                        
                mol_graph = MolGraph(mol)
                if not this_dic['ACSF']:
                    molgraphs['x'] = torch.FloatTensor(mol_graph.f_atoms)
                else:      
                    if not os.path.exists(os.path.join(args.data_path, 'FreeSolv/xyz', this_dic['xyz'], '{}.xyz'.format(idx))):
                        continue
                    atoms = ase_read(os.path.join(args.data_path, 'FreeSolv/xyz', this_dic['xyz'], '{}.xyz'.format(idx)))
                    molgraphs['x'] = torch.FloatTensor(acsf.create(atoms, positions=range(mol.GetNumAtoms()))) # replace the 2D feature vector for atoms

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
                
                style = '3D' if this_dic['ACSF'] else '2D'
                if not os.path.exists(os.path.join(this_dic['save_path'], args.dataset, style, args.xyz, 'graphs/raw')):
                    os.makedirs(os.path.join(this_dic['save_path'], args.dataset, style, args.xyz, 'grapgs/raw'))
                torch.save(examples, os.path.join(this_dic['save_path'], args.dataset, style, args.xyz, 'graphs/raw', 'temp.pt')) ###
                print('Finishing processing {} compounds'.format(len(examples)))
        

if __name__ == '__main__':
    main()

