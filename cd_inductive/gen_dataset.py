import sys
sys.path.append('../')

import re
import random

import os
from os import makedirs
from os.path import isdir, isfile

import random
import itertools
from itertools import repeat
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch_geometric.data import Data

from build_dataset.generate_negative import NegativeSampling

import warnings
warnings.filterwarnings('ignore')


chem_col = 'ChemicalID'
dis_col = 'DiseaseID'


def extract_test_chem():
    raw_path = '../build_dataset/raw'
    desc_path = '../build_dataset/processed'
    mol_feat_path = 'cd/processed'
    
    chem_dis_tmp = pd.read_csv(f'{raw_path}/CTD_chemicals_diseases.csv.gz', skiprows = list(range(27))+[28], compression = 'gzip')
    chem_desc = pd.read_csv(f'{desc_path}/description/chemical_description.csv')
    mol_feat = torch.load(f'{mol_feat_path}/graphmvp_embedding')
    mol_feat = pd.DataFrame({
        'ChemicalID': list(mol_feat.keys()),
        'embedding': [True if not torch.equal(x, torch.zeros((1, 300))) else None for x in mol_feat.values()]
    })
    
    mol_ids = mol_feat[mol_feat.embedding.notna()].ChemicalID
    chem_ids1 = chem_dis_tmp[chem_dis_tmp.ChemicalID.isin(mol_ids)].ChemicalID.unique()
    desc_ids = chem_desc[chem_desc.SummarizedDescription.notna()].ChemicalID
    chem_ids2 = chem_dis_tmp[chem_dis_tmp.ChemicalID.isin(desc_ids)].ChemicalID.unique()
    
    chem_ids = np.unique(np.union1d(chem_ids1, chem_ids2))
    chem_dis_tmp = chem_dis_tmp[chem_dis_tmp.ChemicalID.isin(chem_ids)]

    ### delete data which have 'therapeutic' DirectEvidence
    thera_idx = chem_dis_tmp.DirectEvidence == 'therapeutic'
    chem_dis = chem_dis_tmp[~thera_idx]
    
    # mapping of unique chemical
    uniq_chem = chem_dis[chem_col].unique()
    num_test = int(len(uniq_chem) * 0.05)
    
    return chem_dis, num_test


def construct_graph(chem_dis, mode = 'train', dis_map = None):
    ### curated (DirectEvidence: marker/mechanism) edge index between chem-disease
    curated_chem_dis_idx = chem_dis.DirectEvidence == 'marker/mechanism'
    curated_chem_dis = chem_dis[curated_chem_dis_idx][[chem_col, dis_col]]
    dir_dup_num = curated_chem_dis.duplicated(keep = False).sum()
    if dir_dup_num != 0:
        raise ValueError(f'Duplicated direct evidence: {dir_dup_num}')
    else: 
        print(f'Number of duplicated DirectEvidence: {dir_dup_num}')
    
    ### inferred edge index between chem-disease
    # (c, d) pairs which have DirectEvidence and Inferred Relation
    dup_chem_dis_idx = chem_dis[[chem_col, dis_col]].duplicated(keep = False)
    dup_chem_dis = chem_dis[dup_chem_dis_idx]
    dup_dir_chem_dis = dup_chem_dis[~dup_chem_dis.DirectEvidence.isna()][[chem_col, dis_col]]

    # (c, d) pairs which have Inferred Relation and drops duplicate
    inferred_chem_dis_idx = chem_dis.DirectEvidence.isna()
    inferred_chem_dis = chem_dis[inferred_chem_dis_idx][[chem_col, dis_col]]
    inferred_chem_dis = inferred_chem_dis.drop_duplicates()
    # merge dup_dir_chem_dis and drop which duplicated
    inferred_chem_dis = pd.concat([dup_dir_chem_dis, inferred_chem_dis])
    inferred_chem_dis = inferred_chem_dis.drop_duplicates(keep = False)
    
    ### build graph
    # mapping of unique chemical, disease, and gene
    uniq_chem = chem_dis[chem_col].unique()
    chem_map = {name: i for i, name in enumerate(uniq_chem)}

    if dis_map == None:
        uniq_dis = chem_dis[dis_col].unique()
        dis_map = {name: i for i, name in enumerate(uniq_dis)}

    edge_type_map = {
        'chem_curated_dis': 0,
        'chem_inferred_dis': 1
    }

    # mapping the chemical and disease id
    curated_chem_dis[chem_col] = curated_chem_dis[chem_col].apply(lambda x: chem_map[x])
    curated_chem_dis[dis_col] = curated_chem_dis[dis_col].apply(lambda x: dis_map[x])

    inferred_chem_dis[chem_col] = inferred_chem_dis[chem_col].apply(lambda x: chem_map[x])
    inferred_chem_dis[dis_col] = inferred_chem_dis[dis_col].apply(lambda x: dis_map[x])

    data = Data()
    data.num_nodes_dict = {
        'chemical': len(chem_map),
        'disease': len(dis_map)
    }
    data.edge_index_dict = {
        ('chemical', 'chem_curated_dis', 'disease'): torch.from_numpy(curated_chem_dis.values.T).to(torch.long),
        ('chemical', 'chem_inferred_dis', 'disease'): torch.from_numpy(inferred_chem_dis.values.T).to(torch.long),
    }
    data.edge_reltype = {
        (h, r, t): torch.full((edge.size(1), 1), fill_value = edge_type_map[r]).to(torch.long) for (h ,r ,t), edge in data.edge_index_dict.items()
    }
    data.num_relations = len(edge_type_map)
    
    ### save chemical/disease/rel_type mapping
    save_path = 'dataset/cd/processed'
    
    if isdir(save_path):
        pass
    else:
        makedirs(save_path)
    
    if mode == 'test':
        torch.save(data, f'{save_path}/{mode}_cd.pt')
    else:
        torch.save(data, f'{save_path}/cd.pt')
    torch.save(chem_map, f'{save_path}/{mode}_chem_map')
    torch.save(dis_map, f'{save_path}/{mode}_dis_map')
    torch.save(edge_type_map, f'{save_path}/{mode}_rel_type_map')
    
    if mode == 'train':
        return data, dis_map
    else:
        return data


def build_cd_graph(save_path = 'dataset/cd/processed'):
    ''' Build a Chemical-Disease interaction graph '''
    
    print('----------------------------------------------------------------------------')
    print('>>> Processing Chemical-Disease Data ...')
    
    chem_dis, num_test = extract_test_chem()
    
    uniq_dis = chem_dis[dis_col].unique()
    dis_map = {name: i for i, name in enumerate(uniq_dis)}
    
    random.seed(42)
    test_chem = random.sample(set(chem_dis.ChemicalID.unique()), num_test)
    train_chem = [x for x in set(chem_dis.ChemicalID.unique()) if x not in test_chem]
    
    train_chem_dis = chem_dis[chem_dis.ChemicalID.isin(train_chem)]
    test_chem_dis = chem_dis[chem_dis.ChemicalID.isin(test_chem)]
    
    data, dis_map = construct_graph(train_chem_dis, mode = 'train', dis_map = dis_map)
    test_data = construct_graph(test_chem_dis, mode = 'test', dis_map = dis_map)
    
    random.seed(42)
    train_frac = 0.9 
    valid_frac = 0.05
    
    all_idx = {
        k: list(range(v.shape[1])) for k, v in data.edge_index_dict.items()
    }
    for k, v in all_idx.items():
        random.shuffle(v)

    train_idx = {
        k: all_idx[k][:int(train_frac * len(v))] for k, v in all_idx.items()
    }
    valid_idx = {
        k: all_idx[k][int(train_frac * len(v)) :] for k, v in all_idx.items()
    }
    
    rel_types = list(data.edge_index_dict.keys())

    train_data = Data()
    train_data.num_nodes_dict = data.num_nodes_dict
    train_data.edge_index_dict = {
        k: v[:, train_idx[k]] for k, v in data.edge_index_dict.items()
    }
    train_data.edge_reltype = {
        k: v[train_idx[k]] for k, v in data.edge_reltype.items()
    }
    train_data.num_relations = len(data.edge_index_dict)    
    
    valid_data = Data()
    valid_data.num_nodes_dict = data.num_nodes_dict
    valid_data.edge_index_dict = {
        k: v[:, valid_idx[k]] for k, v in data.edge_index_dict.items()
    }
    valid_data.edge_reltype = {
        k: v[valid_idx[k]] for k, v in data.edge_reltype.items()
    }
    valid_data.num_relations = len(data.edge_index_dict)
    
    print('Chemical-Disease graph is successfully constructed.')
    print('----------------------------------------------------------------------------')
    
    return data, train_data, valid_data, test_data


def negative_sampling(data, all_true_head, all_true_tail, num_negative = 500):
    torch.manual_seed(42)
    
    head_type = []
    tail_type = []
    relation = []
    head = []
    tail = []
    
    for (h, r, t), e in data.edge_index_dict.items():
        head_type.append(list(repeat(h, e.shape[1])))
        tail_type.append(list(repeat(t, e.shape[1])))
        relation.append(data.edge_reltype[(h, r, t)].view(-1))
        head.append(e[0])
        tail.append(e[1])

    triples = {
        'head_type': list(itertools.chain(*head_type)),
        'head': torch.cat(head),
        'relation': torch.cat(relation),
        'tail_type': list(itertools.chain(*tail_type)),
        'tail': torch.cat(tail)
    }

    head_negative_sampling = NegativeSampling(triples, num_negative, 'head-batch', all_true_head, all_true_tail, data.num_nodes_dict)
    head_negative = []
    for i in tqdm(range(len(triples['head']))):
        head_negative.append(head_negative_sampling.__getitem__(i)[1])

    tail_negative_sampling = NegativeSampling(triples, num_negative, 'tail-batch', all_true_head, all_true_tail, data.num_nodes_dict)
    tail_negative = []
    for i in tqdm(range(len(triples['tail']))):
        tail_negative.append(tail_negative_sampling.__getitem__(i)[1])

    head_negative = torch.stack(head_negative)
    tail_negative = torch.stack(tail_negative)
    
    neg_idx_dict = dict()
    cur_idx = 0
    for key in data.edge_index_dict:
        neg_idx_dict[key] = (cur_idx, cur_idx + data.edge_index_dict[key].shape[1])
        cur_idx += data.edge_index_dict[key].shape[1]
    
    data['head_neg'] = {key: head_negative[v[0]:v[1]] for key, v in neg_idx_dict.items()}
    data['tail_neg'] = {key: tail_negative[v[0]:v[1]] for key, v in neg_idx_dict.items()}
    
    return data


def cd_embedding_data():
    data_path = 'dataset/cd/processed'
    
    data = torch.load(f'{data_path}/train_cd.pt')
    chem_map = torch.load(f'{data_path}/train_chem_map')
    dis_map = torch.load(f'{data_path}/train_dis_map')
    rel_map = torch.load(f'{data_path}/train_rel_type_map')
    
    test_data = torch.load(f'{data_path}/test_cd.pt')
    test_chem_map = torch.load(f'{data_path}/test_chem_map')
    test_dis_map = torch.load(f'{data_path}/test_dis_map')
    test_rel_map = torch.load(f'{data_path}/test_rel_type_map')
    
    desc_path = '../build_dataset/processed/description_embedding'
    chem_emb = torch.load(f'{desc_path}/mean_pooling/biot5+_chemical_embedding')
    dis_emb = torch.load(f'{desc_path}/mean_pooling/biot5+_disease_embedding') 
    rel_emb = torch.load(f'{desc_path}/mean_pooling/biot5+_relation_embedding')
    
    mol_feat = torch.load('graphmvp/graphmvp_embedding')
    
    #
    entity_dict = dict()
    cur_idx = 0
    for key in data['num_nodes_dict']:
        entity_dict[key] = (cur_idx, cur_idx + data['num_nodes_dict'][key])
        cur_idx += data['num_nodes_dict'][key]

    chem_map = {k: v + entity_dict['chemical'][0] for k, v in chem_map.items()}
    dis_map = {k: v + entity_dict['disease'][0] for k, v in dis_map.items()}

    text_chem_emb = {k: chem_emb[k] for k in chem_map.keys()}
    chem_emb_tensor = torch.stack([v['text embedding'] for v in text_chem_emb.values()])
    dis_emb = {k: dis_emb[k] for k in dis_map.keys()}
    dis_emb_tensor = torch.stack([v['text embedding'] for v in dis_emb.values()])
    
    entity_embedding = torch.cat([chem_emb_tensor, dis_emb_tensor], dim = 0)
    relation_embedding = torch.stack([rel_emb[k]['text embedding'] for k in rel_map.keys()])
    
    mol_emb = {k: mol_feat[k] for k in chem_map.keys()}
    mol_emb_tensor = torch.cat([v for v in mol_emb.values()], dim = 0)
    
    torch.save(entity_embedding, 'dataset/cd/processed/biot5+_entity_embedding')
    torch.save(relation_embedding, 'dataset/cd/processed/biot5+_relation_embedding')
    torch.save(mol_emb_tensor, 'dataset/cd/processed/molecule_embedding')
    
    
    test_entity_dict = dict()
    cur_idx = 0
    for key in test_data['num_nodes_dict']:
        test_entity_dict[key] = (cur_idx, cur_idx + test_data['num_nodes_dict'][key])
        cur_idx += test_data['num_nodes_dict'][key]

    test_chem_map = {k: v + test_entity_dict['chemical'][0] for k, v in test_chem_map.items()}
    # test_dis_map = {k: v + test_entity_dict['disease'][0] for k, v in test_dis_map.items()}

    test_chem_emb = {k: chem_emb[k] for k in chem_map.keys()}
    test_chem_emb_tensor = torch.stack([v['text embedding'] for v in test_chem_emb.values()])
    # dis_emb = {k: dis_emb[k] for k in dis_map.keys()}
    # dis_emb_tensor = torch.stack([v['text embedding'] for v in dis_emb.values()])
    
    test_entity_embedding = chem_emb_tensor
    # entity_embedding = torch.cat([chem_emb_tensor, dis_emb_tensor], dim = 0)
    # test_relation_embedding = torch.stack([rel_emb[k]['text embedding'] for k in rel_map.keys()])
    
    test_mol_emb = {k: mol_feat[k] for k in test_chem_map.keys()}
    test_mol_emb_tensor = torch.cat([v for v in test_mol_emb.values()], dim = 0)
    
    torch.save(test_entity_embedding, 'dataset/cd/processed/test_biot5+_chemical_embedding')
    torch.save(test_mol_emb_tensor, 'dataset/cd/processed/test_molecule_embedding')


def build_all_triplets(data):
    print('Build all true triplets')
    head_type = []
    tail_type = []
    relation = []
    head = []
    tail = []
    
    for (h, r, t), e in data.edge_index_dict.items():
        head_type.append(list(repeat(h, e.shape[1])))
        tail_type.append(list(repeat(t, e.shape[1])))
        relation.append(data.edge_reltype[(h, r, t)].view(-1))
        head.append(e[0])
        tail.append(e[1])

    triples = {
        'head_type': list(itertools.chain(*head_type)),
        'head': torch.cat(head),
        'relation': torch.cat(relation),
        'tail_type': list(itertools.chain(*tail_type)),
        'tail': torch.cat(tail)
    }

    all_true_head, all_true_tail = defaultdict(list), defaultdict(list)
    for i in tqdm(range(len(triples['head']))):
        head, relation, tail = triples['head'][i].item(), triples['relation'][i].item(), triples['tail'][i].item()
        all_true_head[(relation, tail)].append(head)
        all_true_tail[(head, relation)].append(tail)
    
    return all_true_head, all_true_tail


def build_benchmarks(save_path = 'dataset/cd'):
    print('>>> Build Benchmark Dataset ...')
    
    ### create data
    data, train_data, valid_data, test_data = build_cd_graph()
    
    all_true_head, all_true_tail = build_all_triplets(data)
    test_true_head, test_true_tail = build_all_triplets(test_data)
    
    print('Negative sampling for validation data')
    valid_data = negative_sampling(valid_data, all_true_head, all_true_tail)
    print('Negative sampling for test data')
    test_data = negative_sampling(test_data, test_true_head, test_true_tail)

    ### save splitted data
    torch.save(train_data, f'{save_path}/train_cd.pt')
    torch.save(valid_data, f'{save_path}/valid_cd.pt')
    torch.save(test_data, f'{save_path}/test_cd.pt')
    
    print('Graph construction is completed!')


if __name__ == '__main__':
    build_benchmarks()
    cd_embedding_data()
