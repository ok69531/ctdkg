import os
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
from torch.utils import data
from torch_geometric.data import (Data, InMemoryDataset, download_url, extract_zip)

from config import args
from models import GNN, GNN_graphpred


allowable_features = {
    'possible_atomic_num_list':       list(range(1, 119)),
    'possible_formal_charge_list':    [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
    'possible_chirality_list':        [
        Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        Chem.rdchem.ChiralType.CHI_OTHER
    ],
    'possible_hybridization_list':    [
        Chem.rdchem.HybridizationType.S,
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
        Chem.rdchem.HybridizationType.UNSPECIFIED
    ],
    'possible_numH_list':             [0, 1, 2, 3, 4, 5, 6, 7, 8],
    'possible_implicit_valence_list': [0, 1, 2, 3, 4, 5, 6],
    'possible_degree_list':           [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'possible_bonds':                 [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC
    ],
    'possible_bond_dirs':             [  # only for double bond stereo information
        Chem.rdchem.BondDir.NONE,
        Chem.rdchem.BondDir.ENDUPRIGHT,
        Chem.rdchem.BondDir.ENDDOWNRIGHT
    ]
}


def mol_to_graph_data_obj_simple(mol):
    """ used in MoleculeDataset() class
    Converts rdkit mol objects to graph data object in pytorch geometric
    NB: Uses simplified atom and bond features, and represent as indices
    :param mol: rdkit mol object
    :return: graph data object with the attributes: x, edge_index, edge_attr """

    # atoms
    # num_atom_features = 2  # atom type, chirality tag
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_feature = [allowable_features['possible_atomic_num_list'].index(atom.GetAtomicNum())] + \
                       [allowable_features['possible_chirality_list'].index(atom.GetChiralTag())]
        atom_features_list.append(atom_feature)
    x = torch.tensor(np.array(atom_features_list), dtype=torch.long)

    # bonds
    if len(mol.GetBonds()) <= 0:  # mol has no bonds
        num_bond_features = 2  # bond type & direction
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, num_bond_features), dtype=torch.long)
    else:  # mol has bonds
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_feature = [allowable_features['possible_bonds'].index(bond.GetBondType())] + \
                           [allowable_features['possible_bond_dirs'].index(bond.GetBondDir())]
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)

        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = torch.tensor(np.array(edge_features_list), dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    return data


def main():
    args.input_model_file = 'saved_model/3D_hybrid_03_masking/GEOM_3D_nmol50000_nconf5_nupper1000/CL_1_VAE_1_CP_0.1/6_51_10_0.1/0.15_EBM_dot_prod_0.2_normalize_l2_detach_target_2_100_0/pretraining_model.pth'
    molecule_model = GNN(num_layer = args.num_layer, emb_dim = args.emb_dim,
                        JK = args.JK, drop_ratio = args.dropout_ratio,
                        gnn_type = args.gnn_type)
    molecule_model.load_state_dict(torch.load(args.input_model_file, map_location = 'cpu'))
    molecule_model.eval()
    model = GNN_graphpred(args=args, num_tasks=1, molecule_model=molecule_model)

    chem = pd.read_csv('../../build_dataset/raw/CTD_chemicals_with_smiles.csv')
    chem_emb = {x.split(':')[1]: None for x in chem.ChemicalID}

    for i in tqdm(range(len(chem))):
        chem_id = chem['ChemicalID'][i].split(':')[1]
        smiles = chem['SMILES'][i]
        
        try:
            mol = AllChem.MolFromSmiles(smiles)
            if mol is not None:
                mol_data = mol_to_graph_data_obj_simple(mol)
                with torch.no_grad():
                    mol_features = molecule_model(mol_data)
                mol_features = model.pool(mol_features, torch.zeros(mol_features.size(0)).to(torch.long))
        
        except:
            mol_features = torch.zeros((1, args.emb_dim))
        
        chem_emb[chem_id] = mol_features
    
    save_path = '../dataset/cd/processed'
    if os.path.exists(save_path):
        pass
    else:
        os.makedirs(save_path)
    
    torch.save(chem_emb, os.path.join(save_path, 'graphmvp_embedding'))


if __name__ == '__main__':
    main()
