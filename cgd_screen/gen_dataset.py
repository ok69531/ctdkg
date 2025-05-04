import sys
sys.path.append('../')

import re
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

import warnings
warnings.filterwarnings('ignore')


chem_col = 'ChemicalID'
dis_col = 'DiseaseID'
gene_col = 'GeneID'
go_col = 'GOID'

chem_list = ['D014260', 'C015699', 'C471071', 'C023036', 'C007379', 'C006780', 
             'D002084', 'C041594', 'C017822', 'C029216']
# chem_list = ['D014260', 'C015699', 'C471071', 'C023036', 'C007379', 'C006780', 
#  'D002084', 'C041594', 'C017822', 'C029216', 'D003993', 'C012125', 'C076994']


def split_data(data, train_frac, valid_frac):
    if train_frac + valid_frac*2 != 1:
        raise ValueError('Total Sum != 1')
    
    random.seed(42)

    all_idx = {
        k: list(range(v.shape[1])) for k, v in data.edge_index_dict.items()
    }
    for k, v in all_idx.items():
        random.shuffle(v)

    train_idx = {
        k: all_idx[k][:int(train_frac * len(v))] for k, v in all_idx.items()
    }
    valid_idx = {
        k: all_idx[k][int(train_frac * len(v)) : int(train_frac * len(v)) + int(valid_frac * len(v))] for k, v in all_idx.items()
    }
    test_idx = {
        k: all_idx[k][int(train_frac * len(v)) + int(valid_frac * len(v)):] for k, v in all_idx.items()
    }
    
    rel_types = list(data.edge_index_dict.keys())
    for i in range(len(rel_types)):
        assert len(train_idx[rel_types[i]]) + len(valid_idx[rel_types[i]]) + len(test_idx[rel_types[i]]) == data.edge_index_dict[rel_types[i]].shape[1]

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
    
    test_data = Data()
    test_data.num_nodes_dict = data.num_nodes_dict
    test_data.edge_index_dict = {
        k: v[:, test_idx[k]] for k, v in data.edge_index_dict.items()
    }
    test_data.edge_reltype = {
        k: v[test_idx[k]] for k, v in data.edge_reltype.items()
    }
    test_data.num_relations = len(data.edge_index_dict)

    return train_data, valid_data, test_data


def negative_sampling(data, inferred_gene_dis):
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

    # uniq_head = torch.unique(triples['head'])
    
    tail_negative = []
    for i in range(len(triples['head'])):
        h = triples['head'][i].item()
        
        neg_for_t = inferred_gene_dis[inferred_gene_dis.GeneID == h].DiseaseID
        neg_for_t = torch.from_numpy(neg_for_t.values.T).to(torch.long)
        tail_negative.append(neg_for_t)
    
    # head_negative = []
    # for i in range(len(triples['tail'])):
    #     t = triples['tail'][i].item()
        
    #     neg_for_h = inferred_gene_dis[inferred_gene_dis.DiseaseID == t].GeneID
    #     neg_for_h = torch.from_numpy(neg_for_h.values.T).to(torch.long)
    #     print(neg_for_h.shape)
    #     head_negative.append(neg_for_h)
    
    data['tail_neg'] = {k: torch.nested.nested_tensor(tail_negative) for k in data.edge_index_dict}
    
    return data


def build_cd_graph(save_path = 'dataset/cd'):
    ''' Build a Chemical-Disease interaction graph '''
    
    print('----------------------------------------------------------------------------')
    print('>>> Processing Chemical-Disease Data ...')
    
    raw_path = '../build_dataset/raw'
    desc_path = '../build_dataset/processed/description/chemical_description.csv'
    
    chem_dis_tmp = pd.read_csv(f'{raw_path}/CTD_chemicals_diseases.csv.gz', skiprows = list(range(27))+[28], compression = 'gzip')
    chem_dis_tmp = chem_dis_tmp[chem_dis_tmp[chem_col].isin(chem_list)]
    
    ### delete data which have 'therapeutic' DirectEvidence
    thera_idx = chem_dis_tmp.DirectEvidence == 'therapeutic'
    chem_dis = chem_dis_tmp[~thera_idx]
    
    cd_chem = chem_dis.ChemicalID.unique()
    
    chem_desc_tmp = pd.read_csv(desc_path)
    chem_desc = chem_desc_tmp[chem_desc_tmp.ChemicalID.isin(cd_chem)]
    uniq_chem = chem_desc[chem_desc.SummarizedDescription.notna()].ChemicalID.unique()
    
    chem_dis = chem_dis[chem_dis.ChemicalID.isin(uniq_chem)]
    
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
    if isdir(save_path):
        pass
    else:
        makedirs(save_path)
    
    torch.save(data, f'{save_path}/cd.pt')
    torch.save(chem_map, f'{save_path}/chem_map')
    torch.save(dis_map, f'{save_path}/dis_map')
    torch.save(edge_type_map, f'{save_path}/rel_type_map')
    
    print('Chemical-Disease graph is successfully constructed.')
    print('----------------------------------------------------------------------------')
    
    # return data, save_path


def build_cg_graph(save_path = 'dataset/cg'):
    ''' Build a Chemical-Gene interaction graph '''
    
    print('----------------------------------------------------------------------------')
    print(f'>>> Processing Chemical-Gene Data ...')
    
    file_path = '../build_dataset/raw'
    desc_path = '../build_dataset/processed/description/gene_description.csv'
    
    chem_gene_tmp = pd.read_csv(f'{file_path}/CTD_chem_gene_ixns.csv.gz', skiprows = list(range(27)) + [28], compression = 'gzip')
    chem_map = torch.load('dataset/cd/chem_map')
    
    chem_idx = chem_gene_tmp.ChemicalID.isin(chem_map.keys())
    chem_gene_tmp = chem_gene_tmp[chem_idx]
    
    cg_gene = chem_gene_tmp.GeneID.unique()
    
    gene_desc_tmp = pd.read_csv(desc_path)
    gene_desc = gene_desc_tmp[gene_desc_tmp.GeneID.isin(cg_gene)]
    uniq_gene = gene_desc[gene_desc.SummarizedDescription.notna()].GeneID.unique()
    
    chem_gene_tmp = chem_gene_tmp[chem_gene_tmp.GeneID.isin(uniq_gene)]
    
    # delete data which are not specified the organism
    org_na_idx = chem_gene_tmp['Organism'].isna()
    geneform_na_idx = chem_gene_tmp['GeneForms'].isna()
    chem_gene = chem_gene_tmp[~(org_na_idx|geneform_na_idx)]

    df = chem_gene[[chem_col, gene_col]]
    df['InteractionActions'] = chem_gene['InteractionActions'].map(lambda x: x.split('|'))
    df = df.explode('InteractionActions').reset_index(drop = True).drop_duplicates()
    df['InteractionActions'] = df['InteractionActions'].map(lambda x: re.sub(' ', '-', x))
    
    # mapping
    uniq_rel = df['InteractionActions'].unique()
    edge_type_map = {f'chem_{rel}_gene': i for i, rel in enumerate(uniq_rel)}
    
    uniq_chem = df[chem_col].unique()
    chem_map = {name: i for i, name in enumerate(uniq_chem)}
    
    uniq_gene = df[gene_col].unique()
    gene_map = {name: i for i, name in enumerate(uniq_gene)}
    
    df[chem_col] = df[chem_col].apply(lambda x: chem_map[x])
    df[gene_col] = df[gene_col].apply(lambda x: gene_map[x])
    
    rel_type_df_dict = {
        rel: df[df['InteractionActions'] == rel][[chem_col, gene_col]] for rel in uniq_rel
    }
    
    # data construction
    data = Data()
    data.num_nodes_dict = {
        'chemical': len(chem_map),
        'gene': len(gene_map)
    }
    data.edge_index_dict = {
        ('chemical', f'chem_{rel}_gene', 'gene'): torch.from_numpy(rel_type_df_dict[rel].values.T).to(torch.long) for rel in uniq_rel
    }
    data.edge_reltype = {
        (h, r, t): torch.full((edge.size(1), 1), fill_value=edge_type_map[r]).to(torch.long) for (h, r, t), edge in data.edge_index_dict.items()
    }
    data.num_relations = len(edge_type_map)
    
    ### save mapping
    if isdir(save_path):
        pass
    else:
        makedirs(save_path)
    
    torch.save(data, f'{save_path}/cg.pt')
    torch.save(chem_map, f'{save_path}/chem_map')
    torch.save(gene_map, f'{save_path}/gene_map')
    torch.save(edge_type_map, f'{save_path}/rel_type_map')
        
    print('Chemical-Gene Graph is successfully constructed.')
    print('----------------------------------------------------------------------------')
    
    # return data, save_path


def build_gd_graph(save_path = 'dataset/gd'):
    print('----------------------------------------------------------------------------')
    print('>>> Processing Gene-Disease Data ...')
    print('>>> This procedure may be time-consuming ...')
    
    raw_path = '../build_dataset/raw'
    desc_path = '../build_dataset/processed/description/disease_description.csv'
    
    gene_dis_tmp = pd.read_csv(f'{raw_path}/CTD_genes_diseases.csv.gz', skiprows = list(range(27))+[28], compression = 'gzip')
    gene_map = torch.load('dataset/cg/gene_map')
    
    gene_idx = gene_dis_tmp.GeneID.isin(gene_map.keys())
    gene_dis_tmp = gene_dis_tmp[gene_idx]
    
    gd_dis = gene_dis_tmp.DiseaseID.unique()
    
    dis_desc_tmp = pd.read_csv(desc_path)
    dis_desc = dis_desc_tmp[dis_desc_tmp.DiseaseID.isin(gd_dis)]
    uniq_dis = dis_desc[dis_desc.SummarizedDescription.notna()].DiseaseID.unique()
    
    gene_dis_tmp = gene_dis_tmp[gene_dis_tmp.DiseaseID.isin(uniq_dis)]

    ### delete data which have 'therapeutic' DirectEvidence
    thera_idx = gene_dis_tmp.DirectEvidence == 'therapeutic'
    gene_dis = gene_dis_tmp[~thera_idx]

    ### curated (DirectEvidence: marker/mechanism) edge index between gene-disease
    curated_gene_dis_idx = (gene_dis.DirectEvidence == 'marker/mechanism')|(gene_dis.DirectEvidence == 'marker/mechanism|therapeutic')
    curated_gene_dis = gene_dis[curated_gene_dis_idx][[gene_col, dis_col]]
    dir_dup_num = curated_gene_dis.duplicated(keep = False).sum()
    if dir_dup_num != 0:
        raise ValueError(f'Duplicated direct evidence: {dir_dup_num}')
    else: 
        print(f'Number of duplicated DirectEvidence: {dir_dup_num}')
    
    ## inferred edge index between chem-disease
    # (c, d) pairs which have DirectEvidence and Inferred Relation
    dup_gene_dis_idx = gene_dis[[gene_col, dis_col]].duplicated(keep = False)
    dup_gene_dis = gene_dis[dup_gene_dis_idx]
    dup_dir_gene_dis = dup_gene_dis[~dup_gene_dis.DirectEvidence.isna()][[gene_col, dis_col]]

    # (c, d) pairs which have Inferred Relation and drops duplicate
    inferred_gene_dis_idx = gene_dis.DirectEvidence.isna()
    inferred_gene_dis = gene_dis[inferred_gene_dis_idx][[gene_col, dis_col]]
    inferred_gene_dis = inferred_gene_dis.drop_duplicates()
    # merge dup_dir_gene_dis and drop which duplicated
    inferred_gene_dis = pd.concat([dup_dir_gene_dis, inferred_gene_dis])
    inferred_gene_dis = inferred_gene_dis.drop_duplicates(keep = False)
    
    ### build graph
    # mapping of unique chemical, disease, and gene
    uniq_gene = gene_dis[gene_col].unique()
    gene_map = {name: i for i, name in enumerate(uniq_gene)}

    uniq_dis = gene_dis[dis_col].unique()
    dis_map = {name: i for i, name in enumerate(uniq_dis)}

    edge_type_map = {
        'gene_curated_dis': 0,
        # 'gene_inferred_dis': 1
    }

    # mapping the chemical and disease id
    curated_gene_dis[gene_col] = curated_gene_dis[gene_col].apply(lambda x: gene_map[x])
    curated_gene_dis[dis_col] = curated_gene_dis[dis_col].apply(lambda x: dis_map[x])

    # inferred_gene_dis[gene_col] = inferred_gene_dis[gene_col].apply(lambda x: gene_map[x])
    # inferred_gene_dis[dis_col] = inferred_gene_dis[dis_col].apply(lambda x: dis_map[x])

    data = Data()
    data.num_nodes_dict = {
        'gene': len(gene_map),
        'disease': len(dis_map)
    }
    data.edge_index_dict = {
        ('gene', 'gene_curated_dis', 'disease'): torch.from_numpy(curated_gene_dis.values.T).to(torch.long)
        # ('gene', 'gene_inferred_dis', 'disease'): torch.from_numpy(inferred_gene_dis.values.T).to(torch.long),
    }
    data.edge_reltype = {
        (h, r, t): torch.full((edge.size(1), 1), fill_value = edge_type_map[r]).to(torch.long) for (h, r, t), edge in data.edge_index_dict.items()
    }
    data.num_relations = len(edge_type_map)
    
    ### save chemical/disease/rel_type mapping
    if isdir(save_path):
        pass
    else:
        makedirs(save_path)
    
    torch.save(data, f'{save_path}/gd.pt')
    torch.save(gene_map, f'{save_path}/gene_map')
    torch.save(dis_map, f'{save_path}/dis_map')
    torch.save(edge_type_map, f'{save_path}/rel_type_map')
    
    print('Gene-Disease graph is successfully constructed.')
    print('----------------------------------------------------------------------------')
    
    return inferred_gene_dis


def build_cgd_graph(inferred_gene_dis, file_path = 'dataset', save_path = 'dataset/cgd'):
    print('----------------------------------------------------------------------------')
    print('>>> Processing Chemical-Gene-Disease Data ...')
    print('>>> This procedure may be time-consuming ...')
    
    # needs: cg, cd, gd
    
    # chemical
    cg_data = torch.load(f'{file_path}/cg/cg.pt')
    cg_chem_map = torch.load(f'{file_path}/cg/chem_map')
    cg_gene_map = torch.load(f'{file_path}/cg/gene_map')
    
    cd_data = torch.load(f'{file_path}/cd/cd.pt')
    cd_chem_map = torch.load(f'{file_path}/cd/chem_map')
    cd_dis_map = torch.load(f'{file_path}/cd/dis_map')
    
    # gene
    gd_data = torch.load(f'{file_path}/gd/gd.pt')
    gd_gene_map = torch.load(f'{file_path}/gd/gene_map')
    gd_dis_map = torch.load(f'{file_path}/gd/dis_map')
    
    # mapping
    uniq_chem = set(list(cg_chem_map.keys()) + list(cd_chem_map.keys()))
    chem_map = {name: i for i, name in enumerate(uniq_chem)}
    
    uniq_gene = set(list(cg_gene_map.keys()) + list(gd_gene_map.keys()))
    gene_map = {name: i for i, name in enumerate(uniq_gene)}
    
    uniq_dis = set(list(cd_dis_map.keys()) + list(gd_dis_map.keys()))
    dis_map = {name: i for i, name in enumerate(uniq_dis)}
    
    data_list = [cg_data, cd_data, gd_data]
    rel_type_list = [list(data.edge_index_dict.keys()) for data in data_list]
    rel_type_list = set([r for (h, r, t) in itertools.chain(*rel_type_list)])
    edge_type_map = {r: i for i, r in enumerate(rel_type_list)}
    
    # data construction
    data = Data()
    data.num_nodes_dict = {
        'chemical': len(chem_map),
        'gene': len(gene_map),
        'disease': len(dis_map)
    }
    
    edge_index_list = [list(data.edge_index_dict.items()) for data in data_list]
    edge_index_list = set(list(itertools.chain(*edge_index_list)))
    edge_index_dict = {rel: idx for (rel, idx) in edge_index_list}
    
    print('Converting edge_index')
    for (h, r, t) in tqdm(edge_index_dict.keys()):
        if (h == 'chemical') & (t == 'gene'):
            old_hmap = cg_chem_map; old_tmap = cg_gene_map
            new_hmap = chem_map; new_tmap = gene_map
        elif (h == 'chemical') & (t == 'disease'):
            old_hmap = cd_chem_map; old_tmap = cd_dis_map
            new_hmap = chem_map; new_tmap = dis_map
        elif (h == 'gene') & (t == 'disease'):
            old_hmap = gd_gene_map; old_tmap = gd_dis_map
            new_hmap = gene_map; new_tmap = dis_map
            
            inferred_gene_dis[gene_col] = inferred_gene_dis[gene_col].apply(lambda x: new_hmap[x])
            inferred_gene_dis[dis_col] = inferred_gene_dis[dis_col].apply(lambda x: new_tmap[x])
        
        heads, tails = edge_index_dict[(h, r, t)].numpy()
        # mapping id to entity id
        head_ids = np.array(list(old_hmap.keys()))[heads]
        tail_ids = np.array(list(old_tmap.keys()))[tails]
        # entity id to new mapping id
        new_heads = np.array(list(map(lambda x: new_hmap[x], head_ids)))
        new_tails = np.array(list(map(lambda x: new_tmap[x], tail_ids)))
        
        
        edge_index_dict[(h, r, t)] = torch.from_numpy(np.stack([new_heads, new_tails])).to(torch.long)
    
    data.edge_index_dict =  edge_index_dict
    data.edge_reltype = {
        (h ,r, t): torch.full((edge.size(1), 1), fill_value=edge_type_map[r]) for (h, r, t), edge in data.edge_index_dict.items()
    }
    data.num_relations = len(edge_type_map)
    
    ### save chemical/disease/rel_type mapping
    if isdir(f'{save_path}/processed'):
        pass
    else:
        makedirs(f'{save_path}/processed')
    
    torch.save(data, f'{save_path}/processed/cgd.pt')
    torch.save(chem_map, f'{save_path}/processed/chem_map')
    torch.save(gene_map, f'{save_path}/processed/gene_map')
    torch.save(dis_map, f'{save_path}/processed/dis_map')
    torch.save(edge_type_map, f'{save_path}/processed/rel_type_map')
    
    print('Chemical-Gene-Disease graph is successfully constructed.')
    print('----------------------------------------------------------------------------')
    
    return data, save_path, inferred_gene_dis


def build_benchmarks(inferred_gene_dis, train_frac, valid_frac):
    print('>>> Build Benchmark Dataset ...')
    
    ### create data
    data, save_path, inferred_gene_dis = build_cgd_graph(inferred_gene_dis)
    
    gd_data = Data()
    gd_data.edge_index_dict = {key: edge for key, edge in data.edge_index_dict.items() if key[0] == 'gene'}
    gd_data.edge_reltype = {key: rt for key, rt in data.edge_reltype.items() if key[0] == 'gene'}
    gd_data.num_relations = data.num_relations
    gd_data.num_nodes_dict = data.num_nodes_dict
    
    ### split data
    train_data, valid_data, test_data = split_data(gd_data, train_frac, valid_frac)
    
    # train 데이터에 cg합치기    
    gene_keys = [key for key in data.edge_index_dict if key[0] == 'gene' ]
    for key in gene_keys:
        if key in train_data.edge_index_dict:
            data.edge_index_dict[key] = train_data.edge_index_dict[key]
        if key in train_data.edge_reltype:
            data.edge_reltype[key]   = train_data.edge_reltype[key]
    
    valid_data = negative_sampling(valid_data, inferred_gene_dis)
    test_data = negative_sampling(test_data, inferred_gene_dis)

    ### save splitted data
    torch.save(train_data, f'{save_path}/train_cgd.pt')
    torch.save(valid_data, f'{save_path}/valid_cgd.pt')
    torch.save(test_data, f'{save_path}/test_cgd.pt')
    
    print('Graph construction is completed!')


def cgd_embedding_data():
    data_path = 'dataset/cgd'
    
    data = torch.load(f'{data_path}/train_cgd.pt')
    chem_map = torch.load(f'{data_path}/processed/chem_map')
    gene_map = torch.load(f'{data_path}/processed/gene_map')
    dis_map = torch.load(f'{data_path}/processed/dis_map')
    rel_map = torch.load(f'{data_path}/processed/rel_type_map')
    
    desc_path = '../build_dataset/processed/description_embedding'
    chem_emb = torch.load(f'{desc_path}/mean_pooling/biot5+_chemical_embedding')
    gene_emb = torch.load(f'{desc_path}/mean_pooling/biot5+_gene_embedding') 
    dis_emb = torch.load(f'{desc_path}/mean_pooling/biot5+_disease_embedding') 
    rel_emb = torch.load(f'{desc_path}/mean_pooling/biot5+_relation_embedding')
    
    #
    entity_dict = dict()
    cur_idx = 0
    for key in data['num_nodes_dict']:
        entity_dict[key] = (cur_idx, cur_idx + data['num_nodes_dict'][key])
        cur_idx += data['num_nodes_dict'][key]

    chem_map = {k: v + entity_dict['chemical'][0] for k, v in chem_map.items()}
    gene_map = {k: v + entity_dict['gene'][0] for k, v in gene_map.items()}
    dis_map = {k: v + entity_dict['disease'][0] for k, v in dis_map.items()}

    text_chem_emb = {k: chem_emb[k] for k in chem_map.keys()}
    chem_emb_tensor = torch.stack([v['text embedding'] for v in text_chem_emb.values()])
    text_gene_emb = {k: gene_emb[k] for k in gene_map.keys()}
    gene_emb_tensor = torch.stack([v['text embedding'] for v in text_gene_emb.values()])
    dis_emb = {k: dis_emb[k] for k in dis_map.keys()}
    dis_emb_tensor = torch.stack([v['text embedding'] for v in dis_emb.values()])
    
    entity_embedding = torch.cat([chem_emb_tensor, gene_emb_tensor, dis_emb_tensor], dim = 0)
    relation_embedding = torch.stack([rel_emb[k]['text embedding'] for k in rel_map.keys()])
    
    torch.save(entity_embedding, 'dataset/cgd/processed/biot5+_entity_embedding')
    torch.save(relation_embedding, 'dataset/cgd/processed/biot5+_relation_embedding')


if __name__ == '__main__':
    build_cd_graph()
    build_cg_graph()
    inferred_gene_dis = build_gd_graph()
    
    # build_cgd_graph(inferred_gene_dis)
    
    build_benchmarks(inferred_gene_dis, 0.9, 0.05)
