from typing import List, Tuple, Union
from rdkit import Chem
import collections
#from descriptastorus.descriptors import rdDescriptors, rdNormalizedDescriptors

#__all__ = [get_atom_fdim, get_bond_fdim, MolGraph]

######################## Define atom features and bond features ##############
ATOM_FEATURES = {
    'atom_symbol': ['H', 'C', 'N', 'O', 'S', 'F', 'I', 'P', 'Cl', 'Br'], 
    'atom_degree': [0, 1, 2, 3, 4, 5],
    'atom_explicitValence': [0, 1, 2, 3, 4, 5, 6],
    'atom_implicitValence': [0, 1, 2, 3, 4, 5],
    'hybridization': [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2
    ],
}

# allowable node and edge features
BOND_FEATURES = {
    'possible_bonds' : [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC
    ],
    'possible_bond_dirs' : [ # only for double bond stereo information
        Chem.rdchem.BondDir.NONE,
        Chem.rdchem.BondDir.ENDUPRIGHT,
        Chem.rdchem.BondDir.ENDDOWNRIGHT,
        Chem.rdchem.BondDir.EITHERDOUBLE
    ]
}

######################## Define atom features and bond features ##############

##############################################################################
def onek_encoding_unk(value, choices: List[int]) -> List[int]:
    encoding = [0] * (len(choices) + 1)
    index = choices.index(value) if value in choices else -1
    encoding[index] = 1

    return encoding

def get_atom_fdim():
    return sum(len(choices)+1 for choices in ATOM_FEATURES.values()) + 1

def get_bond_fdim():
    return sum(len(choices)+1 for choices in BOND_FEATURES.values()) + 1

def get_atom_features(atom):
    features = onek_encoding_unk(atom.GetSymbol(), ATOM_FEATURES['atom_symbol']) + \
           onek_encoding_unk(atom.GetHybridization(), ATOM_FEATURES['hybridization']) + \
           onek_encoding_unk(atom.GetTotalDegree(), ATOM_FEATURES['atom_degree']) + \
           onek_encoding_unk(atom.GetExplicitValence(), ATOM_FEATURES['atom_explicitValence']) + \
           onek_encoding_unk(atom.GetImplicitValence(), ATOM_FEATURES['atom_implicitValence']) + \
           [1 if atom.GetIsAromatic() else 0] + \
           [1 if atom.IsInRingSize(3) else 0] + \
           [1 if atom.IsInRingSize(4) else 0] + \
           [1 if atom.IsInRingSize(5) else 0] + \
           [1 if atom.IsInRingSize(6) else 0] + \
           [1 if atom.IsInRingSize(7) else 0] + \
           [1 if atom.IsInRingSize(8) else 0] + \
           [1 if atom.IsInRingSize(9) else 0] + \
           [1 if atom.IsInRingSize(10) else 0] + \
           [1 if atom.IsInRingSize(11) else 0] + \
           [1 if atom.IsInRingSize(12) else 0] 
    if atom.IsInRingSize(13) or atom.IsInRingSize(14) or atom.IsInRingSize(15) or atom.IsInRingSize(16) or atom.IsInRingSize(17) or atom.IsInRingSize(18) or atom.IsInRingSize(19) or atom.IsInRingSize(20): features + [1] 
    else: features + [0]
               

    return features


def get_bond_features(bond):
    """
    Builds a feature vector for a bond.
    :param bond: A RDKit bond.
    :return: A list containing the bond features.
    """
    if bond is None:
        fbond = [1] + [0] * (BOND_FDIM - 1)
    else:
        bt = bond.GetBondType()
        fbond = [
            0,  # bond is not None
            bt == Chem.rdchem.BondType.SINGLE,
            bt == Chem.rdchem.BondType.DOUBLE,
            bt == Chem.rdchem.BondType.TRIPLE,
            bt == Chem.rdchem.BondType.AROMATIC,
            (bond.GetIsConjugated() if bt is not None else 0),
             (bond.IsInRing() if bt is not None else 0)
         ] 
    return fbond

class MolGraph:
    """
    A MolGraph represents the graph structure and featurization of a single molecule.
    A MolGraph computes the following attributes:
    - n_atoms: The number of atoms in the molecule.
    - n_bonds: The number of bonds in the molecule.
    - f_atoms: A mapping from an atom index to a list atom features.
    - f_bonds: A mapping from a bond index to a list of bond features.
    """

    def __init__(self, mol):
        """
        Computes the graph structure and featurization of a molecule.
        :param mol: A SMILES string or an RDKit molecule.
        """
        # Convert SMILES to RDKit molecule if necessary
        if type(mol) == str:
            mol = Chem.MolFromSmiles(mol)
        if type(mol) == Chem.rdchem.Mol:
            mol = mol

        self.n_atoms = 0  # number of atoms
        self.n_bonds = 0  # number of bonds
        self.f_atoms = []  # mapping from atom index to atom features
        self.f_bonds = []  # mapping from bond index to concat(in_atom, bond) features
        self.at_begin = []
        self.at_end = []

        # Get atom/bond features
        self.f_atoms = [get_atom_features(atom) for atom in mol.GetAtoms()]
        self.n_atoms = len(self.f_atoms)

        for bond in mol.GetBonds():
            bf = get_bond_features(bond)
            self.f_bonds.append(bf)
            self.f_bonds.append(bf)
            self.n_bonds += 1
                
            self.at_begin.append(bond.GetBeginAtom().GetIdx())
            self.at_begin.append(bond.GetEndAtom().GetIdx())
            self.at_end.append(bond.GetEndAtom().GetIdx())
            self.at_end.append(bond.GetBeginAtom().GetIdx())


class MolGraph_dmpnn:
    
    def __init__(self, mol, ACSF=False, f_atoms=None):
        
        # Convert SMILES to RDKit molecule if necessary
        if type(mol) == str:
            mol = Chem.MolFromSmiles(mol)
        if type(mol) == Chem.rdchem.Mol:
            mol = mol
        
        self.n_atoms = 0  # number of atoms
        self.n_bonds = 0  # number of bonds
        self.f_atoms = []  # mapping from atom index to atom features
        self.f_bonds = []  # mapping from bond index to concat(in_atom, bond) features
        self.a2b = []  # mapping from atom index to incoming bond indices
        self.b2a = []  # mapping from bond index to the index of the atom the bond is coming from
        self.b2revb = []  # mapping from bond index to the index of the reverse bond
        self.at_begin = []
        self.at_end = []

        if ACSF:
            self.f_atoms = f_atoms
        else:
            self.f_atoms = [get_atom_features(atom) for atom in mol.GetAtoms()]
        self.n_atoms = len(self.f_atoms)

        # Initialize atom to bond mapping for each atom
        for _ in range(self.n_atoms):
            self.a2b.append([])

        # Get bond features
        for a1 in range(self.n_atoms):
            for a2 in range(a1 + 1, self.n_atoms):
                bond = mol.GetBondBetweenAtoms(a1, a2)

                if bond is None:
                    continue

                f_bond = get_bond_features(bond)
                self.at_begin.append(bond.GetBeginAtom().GetIdx())
                self.at_begin.append(bond.GetEndAtom().GetIdx())
                self.at_end.append(bond.GetEndAtom().GetIdx())
                self.at_end.append(bond.GetBeginAtom().GetIdx())

                self.f_bonds.append(self.f_atoms[a1] + f_bond)
                self.f_bonds.append(self.f_atoms[a2] + f_bond)

                # Update index mappings
                b1 = self.n_bonds
                b2 = b1 + 1
                self.a2b[a2].append(b1)  # b1 = a1 --> a2
                self.b2a.append(a1)
                self.a2b[a1].append(b2)  # b2 = a2 --> a1
                self.b2a.append(a2)
                self.b2revb.append(b2)
                self.b2revb.append(b1)
                self.n_bonds += 2