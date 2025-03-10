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

from generate_negative import NegativeSampling

import warnings
warnings.filterwarnings('ignore')

chem_col = 'ChemicalID'
dis_col = 'DiseaseID'
gene_col = 'GeneID'
path_col = 'PathwayID'
go_col = 'GOID'


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

    # true_head, true_tail = defaultdict(list), defaultdict(list)
    # for i in tqdm(range(len(triples['head']))):
    #     head, relation, tail = triples['head'][i].item(), triples['relation'][i].item(), triples['tail'][i].item()
    #     head_type, tail_type = triples['head_type'][i], triples['tail_type'][i]
        # true_head[(relation, tail)].append(head)
        # true_tail[(head, relation)].append(tail)

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


def build_cd_graph(file_path = 'raw', save_path = 'processed/cd'):
    ''' Build a Chemical-Disease interaction graph '''
    
    print('>>> Processing Chemical-Disease Data ...')
    print('----------------------------------------------------------------------------')
    
    chem_dis_tmp = pd.read_csv(f'{file_path}/CTD_chemicals_diseases.csv.gz', skiprows = list(range(27))+[28], compression = 'gzip')

    ### delete data which have 'therapeutic' DirectEvidence
    thera_idx = chem_dis_tmp.DirectEvidence == 'therapeutic'
    chem_dis = chem_dis_tmp[~thera_idx]

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
    data.num_relations = len(data.edge_index_dict)
    
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
    
    return data, save_path


def build_cg_graph(file_path = 'raw', save_path = 'processed/cg'):
    ''' Build a Chemical-Gene interaction graph '''
    
    print(f'>>> Processing Chemical-Gene Data ...')
    print('----------------------------------------------------------------------------')
    
    chem_gene_tmp = pd.read_csv(f'{file_path}/CTD_chem_gene_ixns.csv.gz', skiprows = list(range(27)) + [28], compression = 'gzip')

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
    data.num_relations = len(uniq_rel)
    
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
    
    return data, save_path


def build_cpheno_graph(file_path = 'raw', save_path = 'processed/cpheno'):
    print('>>> Processing Chemical-Phenotype Data ...')
    print('----------------------------------------------------------------------------')
    
    chem_pheno_tmp = pd.read_csv(
        f'{file_path}/CTD_pheno_term_ixns.csv.gz',
        skiprows = list(range(27))+[28], compression = 'gzip')
    
    chem_col = 'chemicalid'
    pheno_col = 'phenotypeid'
    
    chem_pheno = chem_pheno_tmp[[chem_col, pheno_col]].drop_duplicates()
    
    ### build graph
    uniq_chem = chem_pheno[chem_col].unique()
    chem_map = {name: i for i, name in enumerate(uniq_chem)}

    uniq_pheno = chem_pheno[pheno_col].unique()
    pheno_map = {name: i for i, name in enumerate(uniq_pheno)}

    edge_type_map = {
        'chem_inferred_pheno': 0
    }

    # mapping the phenotype and disease id
    chem_pheno[chem_col] = chem_pheno[chem_col].apply(lambda x: chem_map[x])
    chem_pheno[pheno_col] = chem_pheno[pheno_col].apply(lambda x: pheno_map[x])

    data = Data()
    data.num_nodes_dict = {
        'chemical': len(chem_map),
        'phenotype': len(pheno_map)
    }
    data.edge_index_dict = {
        ('chemical', 'chem_inferred_pheno', 'phenotype'): torch.from_numpy(chem_pheno.values.T).to(torch.long)
    }
    data.edge_reltype = {
        (h, r, t): torch.full((edge.size(1), 1), fill_value=edge_type_map[r]) for (h, r, t), edge in data.edge_index_dict.items()
    }
    data.num_relations = len(data.edge_index_dict)

    ### save chemical/disease/rel_type mapping
    if isdir(save_path):
        pass
    else:
        makedirs(save_path)

    torch.save(data, f'{save_path}/cpheno.pt')
    torch.save(chem_map, f'{save_path}/chem_map')
    torch.save(pheno_map, f'{save_path}/pheno_map')
    torch.save(edge_type_map, f'{save_path}/rel_type_map')

    print('Chemical-Phenotype graph is successfully constructed.')
    
    return data, save_path


def build_cpath_graph(file_path = 'raw', save_path = 'processed/cpath'):
    print('>>> Processing Chemical-Pathway Data ...')
    print('----------------------------------------------------------------------------')
    
    # this file does not have duplicated (chem, pathway) pair
    chem_path_tmp = pd.read_csv(f'{file_path}/CTD_chem_pathways_enriched.csv.gz', skiprows = list(range(27))+[28], compression = 'gzip')
    chem_path_tmp.PathwayName
    ### build graph
    uniq_chem = chem_path_tmp[chem_col].unique()
    chem_map = {name: i for i, name in enumerate(uniq_chem)}

    uniq_path = chem_path_tmp[path_col].unique()
    path_map = {name: i for i, name in enumerate(uniq_path)}

    edge_type_map = {
        'chem_associated_path': 0
    }

    # mapping the chemical and disease id
    chem_path = chem_path_tmp[[chem_col, path_col]]
    chem_path[chem_col] = chem_path[chem_col].apply(lambda x: chem_map[x])
    chem_path[path_col] = chem_path[path_col].apply(lambda x: path_map[x])

    data = Data()
    data.num_nodes_dict = {
        'chemical': len(chem_map),
        'pathway': len(path_map)
    }
    data.edge_index_dict = {
        ('chemical', 'chem_associated_path', 'pathway'): torch.from_numpy(chem_path.values.T).to(torch.long),
    }
    data.edge_reltype = {
        (h, r, t): torch.full((edge.size(1), 1), fill_value = edge_type_map[r]).to(torch.long) for (h, r, t), edge in data.edge_index_dict.items()
    }
    data.num_relations = len(data.edge_index_dict)
    
    ### save mapping
    if isdir(save_path):
        pass
    else:
        makedirs(save_path)
    
    torch.save(data, f'{save_path}/cpath.pt')
    torch.save(chem_map, f'{save_path}/chem_map')
    torch.save(path_map, f'{save_path}/path_map')
    torch.save(edge_type_map, f'{save_path}/rel_type_map')
    
    print('Chemical-Pathway graph is successfully constructed.')
    
    return data, save_path


def build_cgo_graph(file_path = 'raw', save_path = 'processed/cgo'):
    print('>>> Processing Chemical-GeneOntology Data ...')
    print('----------------------------------------------------------------------------')
    
    chem_go_tmp = pd.read_csv(
        f'{file_path}/CTD_chem_go_enriched.csv.gz',
        skiprows = list(range(27))+[28], compression = 'gzip')

    bio_idx = chem_go_tmp['Ontology'] == 'Biological Process'
    cell_idx = chem_go_tmp['Ontology'] == 'Cellular Component'
    mol_idx = chem_go_tmp['Ontology'] == 'Molecular Function'
    
    go_col = 'GOTermID'
    biological_chem_go = chem_go_tmp[bio_idx][[chem_col, go_col]]
    cellular_chem_go = chem_go_tmp[cell_idx][[chem_col, go_col]]
    molecular_chem_go = chem_go_tmp[mol_idx][[chem_col, go_col]]
    
    # there are not have duplicated (phenotype, disease) pair
    all_chem_go = pd.concat([biological_chem_go, cellular_chem_go, molecular_chem_go])

    ### build graph
    uniq_chem = all_chem_go[chem_col].unique()
    chem_map = {name: i for i, name in enumerate(uniq_chem)}

    uniq_go = all_chem_go[go_col].unique()
    go_map = {name: i for i, name in enumerate(uniq_go)}

    edge_type_map = {
        'chem_inferred_biological_go': 0,
        'chem_inferred_cellular_go': 1,
        'chem_inferred_molecular_go': 2
    }

    # mapping the chemical and gene ontology id
    biological_chem_go[chem_col] = biological_chem_go[chem_col].apply(lambda x: chem_map[x])
    biological_chem_go[go_col] = biological_chem_go[go_col].apply(lambda x: go_map[x])

    cellular_chem_go[chem_col] = cellular_chem_go[chem_col].apply(lambda x: chem_map[x])
    cellular_chem_go[go_col] = cellular_chem_go[go_col].apply(lambda x: go_map[x])

    molecular_chem_go[chem_col] = molecular_chem_go[chem_col].apply(lambda x: chem_map[x])
    molecular_chem_go[go_col] = molecular_chem_go[go_col].apply(lambda x: go_map[x])

    data = Data()
    data.num_nodes_dict = {
        'chemical': len(chem_map),
        'gene_ontology': len(go_map)
    }
    data.edge_index_dict = {
        ('chemical', 'chem_inferred_biological_go', 'gene_ontology'): torch.from_numpy(biological_chem_go.values.T).to(torch.long),
        ('chemical', 'chem_inferred_cellular_go', 'gene_ontology'): torch.from_numpy(cellular_chem_go.values.T).to(torch.long),
        ('chemical', 'chem_inferred_molecular_go', 'gene_ontology'): torch.from_numpy(molecular_chem_go.values.T).to(torch.long)
    }
    data.edge_reltype = {
        (h, r, t): torch.full((edge.size(1), 1), fill_value=edge_type_map[r]) for (h, r, t), edge in data.edge_index_dict.items()
    }
    data.num_relations = len(data.edge_index_dict)

    ### save chemical/disease/rel_type mapping
    if isdir(save_path):
        pass
    else:
        makedirs(save_path)

    torch.save(data, f'{save_path}/cgo.pt')
    torch.save(chem_map, f'{save_path}/chem_map')
    torch.save(go_map, f'{save_path}/go_map')
    torch.save(edge_type_map, f'{save_path}/rel_type_map')

    print('Chemical-GeneOntology graph is successfully constructed.')
    
    return data, save_path


def build_gpath_graph(file_path = 'raw', save_path = 'processed/gpath'):
    print('>>> Processing Gene-Pathway Data ...')
    print('----------------------------------------------------------------------------')
    
    # this file does not have duplicated (chem, pathway) pair
    gene_path_tmp = pd.read_csv(f'{file_path}/CTD_genes_pathways.csv.gz', skiprows = list(range(27))+[28], compression = 'gzip')
    
    ### build graph
    uniq_gene = gene_path_tmp[gene_col].unique()
    gene_map = {name: i for i, name in enumerate(uniq_gene)}

    uniq_path = gene_path_tmp[path_col].unique()
    path_map = {name: i for i, name in enumerate(uniq_path)}

    edge_type_map = {
        'gene_associated_path': 0,
    }

    # mapping the chemical and disease id
    gene_path = gene_path_tmp[[gene_col, path_col]]
    gene_path[gene_col] = gene_path[gene_col].apply(lambda x: gene_map[x])
    gene_path[path_col] = gene_path[path_col].apply(lambda x: path_map[x])

    data = Data()
    data.num_nodes_dict = {
        'gene': len(gene_map),
        'pathway': len(path_map)
    }
    data.edge_index_dict = {
        ('gene', 'gene_associated_path', 'pathway'): torch.from_numpy(gene_path.values.T).to(torch.long),
        ('pathway', 'gene_associated_path', 'gene'): torch.from_numpy(gene_path.values.T[[1,0]]).to(torch.long),
    }
    data.edge_reltype = {
        (h, r, t): torch.full((edge.size(1), 1), fill_value=edge_type_map[r]) for (h, r, t), edge in data.edge_index_dict.items()
    }
    data.num_relations = len(data.edge_index_dict)
    
    ### save mapping
    if isdir(save_path):
        pass
    else:
        makedirs(save_path)
    
    torch.save(data, f'{save_path}/gpath.pt')
    torch.save(gene_map, f'{save_path}/gene_map')
    torch.save(path_map, f'{save_path}/path_map')
    torch.save(edge_type_map, f'{save_path}/rel_type_map')
    
    print('Gene-Pathway graph is successfully constructed.')
    
    return data, save_path


def build_gd_graph(file_path = 'raw', save_path = 'processed/gd'):
    print('This procedure may be time-consuming ...')
    print('>>> Processing Gene-Disease Data ...')
    print('----------------------------------------------------------------------------')
    
    gene_dis_tmp = pd.read_csv(f'{file_path}/CTD_genes_diseases.csv.gz', skiprows = list(range(27))+[28], compression = 'gzip')

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
    
    ### inferred edge index between chem-disease
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
        'gene_inferred_dis': 1
    }

    # mapping the chemical and disease id
    curated_gene_dis[gene_col] = curated_gene_dis[gene_col].apply(lambda x: gene_map[x])
    curated_gene_dis[dis_col] = curated_gene_dis[dis_col].apply(lambda x: dis_map[x])

    inferred_gene_dis[gene_col] = inferred_gene_dis[gene_col].apply(lambda x: gene_map[x])
    inferred_gene_dis[dis_col] = inferred_gene_dis[dis_col].apply(lambda x: dis_map[x])

    data = Data()
    data.num_nodes_dict = {
        'gene': len(gene_map),
        'disease': len(dis_map)
    }
    data.edge_index_dict = {
        ('gene', 'gene_curated_dis', 'disease'): torch.from_numpy(curated_gene_dis.values.T).to(torch.long),
        ('gene', 'gene_inferred_dis', 'disease'): torch.from_numpy(inferred_gene_dis.values.T).to(torch.long),
    }
    data.edge_reltype = {
        (h, r, t): torch.full((edge.size(1), 1), fill_value = edge_type_map[r]).to(torch.long) for (h, r, t), edge in data.edge_index_dict.items()
    }
    data.num_relations = len(data.edge_index_dict)
    
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
    
    return data, save_path


def build_ggo_graph(file_path = 'raw', save_path = 'processed/ggo'):
    print('>>> Processing Gene-GeneOntology Data ...')
    print('----------------------------------------------------------------------------')
    
    biological_gene_go_tmp = pd.read_csv(f'{file_path}/CTD_gene_GO_biological.csv')
    celluar_gene_go_tmp = pd.read_csv(f'{file_path}/CTD_gene_GO_cellular.csv')
    molecular_gene_go_tmp = pd.read_csv(f'{file_path}/CTD_gene_GO_molecular.csv')

    go_col = 'GO Term ID'
    gene_col = 'Gene ID'
    biological_gene_go = biological_gene_go_tmp[[gene_col, go_col]]
    cellular_gene_go = celluar_gene_go_tmp[[gene_col, go_col]]
    molecular_gene_go = molecular_gene_go_tmp[[gene_col, go_col]]
    
    # there are not have duplicated (phenotype, disease) pair
    all_gene_go = pd.concat([biological_gene_go, cellular_gene_go, molecular_gene_go])

    ### build graph
    uniq_gene = all_gene_go[gene_col].unique()
    gene_map = {name: i for i, name in enumerate(uniq_gene)}

    uniq_go = all_gene_go[go_col].unique()
    go_map = {name: i for i, name in enumerate(uniq_go)}

    edge_type_map = {
        'gene_inferred_biological_go': 0,
        'gene_inferred_cellular_go': 1,
        'gene_inferred_molecular_go': 2,
    }

    # mapping the chemical and gene ontology id
    biological_gene_go[gene_col] = biological_gene_go[gene_col].apply(lambda x: gene_map[x])
    biological_gene_go[go_col] = biological_gene_go[go_col].apply(lambda x: go_map[x])

    cellular_gene_go[gene_col] = cellular_gene_go[gene_col].apply(lambda x: gene_map[x])
    cellular_gene_go[go_col] = cellular_gene_go[go_col].apply(lambda x: go_map[x])

    molecular_gene_go[gene_col] = molecular_gene_go[gene_col].apply(lambda x: gene_map[x])
    molecular_gene_go[go_col] = molecular_gene_go[go_col].apply(lambda x: go_map[x])

    data = Data()
    data.num_nodes_dict = {
        'gene': len(gene_map),
        'gene_ontology': len(go_map)
    }
    data.edge_index_dict = {
        ('gene', 'gene_inferred_biological_go', 'gene_ontology'): torch.from_numpy(biological_gene_go.values.T).to(torch.long),
        ('gene_ontology', 'gene_inferred_biological_go', 'gene'): torch.from_numpy(biological_gene_go.values.T[[1,0]]).to(torch.long),
        ('gene', 'gene_inferred_cellular_go', 'gene_ontology'): torch.from_numpy(cellular_gene_go.values.T).to(torch.long),
        ('gene_ontology', 'gene_inferred_cellular_go', 'gene'): torch.from_numpy(cellular_gene_go.values.T[[1,0]]).to(torch.long),
        ('gene', 'gene_inferred_molecular_go', 'gene_ontology'): torch.from_numpy(molecular_gene_go.values.T).to(torch.long),
        ('gene_ontology', 'gene_inferred_molecular_go', 'gene'): torch.from_numpy(molecular_gene_go.values.T[[1,0]]).to(torch.long)
    }
    data.edge_reltype = {
        (h, r, t): torch.full((edge.size(1), 1), fill_value=edge_type_map[r]) for (h, r, t), edge in data.edge_index_dict.items()
    }
    data.num_relations = len(data.edge_index_dict)

    ### save chemical/disease/rel_type mapping
    if isdir(save_path):
        pass
    else:
        makedirs(save_path)

    torch.save(data, f'{save_path}/ggo.pt')
    torch.save(gene_map, f'{save_path}/gene_map')
    torch.save(go_map, f'{save_path}/go_map')
    torch.save(edge_type_map, f'{save_path}/rel_type_map')

    print('Gene-GeneOntology graph is successfully constructed.')
    
    return data, save_path


def build_gpheno_graph(file_path = 'raw/gene_phenotype', save_path = 'processed/gpheno'):
    print('>>> Processing Gene-Phenotype Data ...')
    print('----------------------------------------------------------------------------')
    
    file_list = os.listdir(file_path)
    gene_pheno_tmp = pd.concat([pd.read_csv(f'{file_path}/{f}') for f in tqdm(file_list)])
    
    gene_col = 'Gene ID'
    pheno_col = 'Phenotype ID'
    gene_pheno = gene_pheno_tmp[[gene_col, pheno_col]].drop_duplicates()
    
    ### build graph
    uniq_gene = gene_pheno[gene_col].unique()
    gene_map = {name: i for i, name in enumerate(uniq_gene)}

    uniq_pheno = gene_pheno[pheno_col].unique()
    pheno_map = {name: i for i, name in enumerate(uniq_pheno)}

    edge_type_map = {
        'gene_associated_pheno': 0,
    }

    # mapping the phenotype and disease id
    gene_pheno[gene_col] = gene_pheno[gene_col].apply(lambda x: gene_map[x])
    gene_pheno[pheno_col] = gene_pheno[pheno_col].apply(lambda x: pheno_map[x])

    data = Data()
    data.num_nodes_dict = {
        'gene': len(gene_map),
        'phenotype': len(pheno_map)
    }
    data.edge_index_dict = {
        ('gene', 'gene_associated_pheno', 'phenotype'): torch.from_numpy(gene_pheno.values.T).to(torch.long),
        ('phenotype', 'gene_associated_pheno', 'gene'): torch.from_numpy(gene_pheno.values.T[[1, 0]]).to(torch.long)
    }
    data.edge_reltype = {
        (h, r, t): torch.full((edge.size(1), 1), fill_value=edge_type_map[r]) for (h, r, t), edge in data.edge_index_dict.items()
    }
    data.num_relations = len(data.edge_index_dict)

    ### save chemical/disease/rel_type mapping
    if isdir(save_path):
        pass
    else:
        makedirs(save_path)

    torch.save(data, f'{save_path}/gpheno.pt')
    torch.save(gene_map, f'{save_path}/gene_map')
    torch.save(pheno_map, f'{save_path}/pheno_map')
    torch.save(edge_type_map, f'{save_path}/rel_type_map')

    print('Gene-Phenotype graph is successfully constructed.')
    
    return data, save_path


def build_dpath_graph(file_path = 'raw', save_path = 'processed/dpath'):
    print('>>> Processing Disease-Pathway Data ...')
    print('----------------------------------------------------------------------------')
    
    # this file does not have duplicated (chem, pathway) pair
    dis_path_tmp = pd.read_csv(f'{file_path}/CTD_diseases_pathways.csv.gz', skiprows = list(range(27))+[28], compression = 'gzip')
    
    ### build graph
    uniq_dis = dis_path_tmp[dis_col].unique()
    dis_map = {name: i for i, name in enumerate(uniq_dis)}

    uniq_path = dis_path_tmp[path_col].unique()
    path_map = {name: i for i, name in enumerate(uniq_path)}

    edge_type_map = {
        'dis_associated_path': 0, 
    }

    # mapping the chemical and disease id
    dis_path = dis_path_tmp[[dis_col, path_col]]
    dis_path[dis_col] = dis_path[dis_col].apply(lambda x: dis_map[x])
    dis_path[path_col] = dis_path[path_col].apply(lambda x: path_map[x])

    data = Data()
    data.num_nodes_dict = {
        'disease': len(dis_map),
        'pathway': len(path_map)
    }
    data.edge_index_dict = {
        ('disease', 'dis_associated_path', 'pathway'): torch.from_numpy(dis_path.values.T).to(torch.long),
        ('pathway', 'dis_associated_path', 'disease'): torch.from_numpy(dis_path.values.T[[1,0]]).to(torch.long),
    }
    data.edge_reltype = {
        (h, r, t): torch.full((edge.size(1), 1), fill_value=edge_type_map[r]) for (h, r, t), edge in data.edge_index_dict.items()
    }
    data.num_relations = len(data.edge_index_dict)
    
    ### save mapping
    if isdir(save_path):
        pass
    else:
        makedirs(save_path)

    torch.save(data, f'{save_path}/dpath.pt')
    torch.save(dis_map, f'{save_path}/dis_map')
    torch.save(path_map, f'{save_path}/path_map')
    torch.save(edge_type_map, f'{save_path}/rel_type_map')
    
    print('Disease-Pathway graph is successfully constructed.')
    
    return data, save_path


def build_dpheno_graph(file_path = 'raw', save_path = 'processed/dpheno'):
    print('>>> Processing Phenotype-Disease Data ...')
    print('----------------------------------------------------------------------------')
    biological_pheno_dis_tmp = pd.read_csv(
        f'{file_path}/CTD_Phenotype-Disease_biological_process_associations.csv.gz',
        skiprows = list(range(27))+[28], compression = 'gzip')
    cellular_pheno_dis_tmp = pd.read_csv(
        f'{file_path}/CTD_Phenotype-Disease_cellular_component_associations.csv.gz',
        skiprows = list(range(27))+[28], compression = 'gzip')
    molecular_pheno_dis_tmp = pd.read_csv(
        f'{file_path}/CTD_Phenotype-Disease_molecular_function_associations.csv.gz',
        skiprows = list(range(27))+[28], compression = 'gzip')

    biological_pheno_dis = biological_pheno_dis_tmp[[go_col, dis_col]]
    cellular_pheno_dis = cellular_pheno_dis_tmp[[go_col, dis_col]]
    molecular_pheno_dis = molecular_pheno_dis_tmp[[go_col, dis_col]]

    # there are not have duplicated (phenotype, disease) pair
    all_pheno_dis = pd.concat([biological_pheno_dis, cellular_pheno_dis, molecular_pheno_dis])

    ### build graph
    uniq_pheno = all_pheno_dis[go_col].unique()
    pheno_map = {name: i for i, name in enumerate(uniq_pheno)}

    uniq_dis = all_pheno_dis[dis_col].unique()
    dis_map = {name: i for i, name in enumerate(uniq_dis)}

    edge_type_map = {
        'dis_associated_biological_pheno': 0,
        'dis_associated_cellular_pheno': 1,
        'dis_associated_molecular_pheno': 2,
    }

    # mapping the phenotype and disease id
    biological_pheno_dis[go_col] = biological_pheno_dis[go_col].apply(lambda x: pheno_map[x])
    biological_pheno_dis[dis_col] = biological_pheno_dis[dis_col].apply(lambda x: dis_map[x])

    cellular_pheno_dis[go_col] = cellular_pheno_dis[go_col].apply(lambda x: pheno_map[x])
    cellular_pheno_dis[dis_col] = cellular_pheno_dis[dis_col].apply(lambda x: dis_map[x])

    molecular_pheno_dis[go_col] = molecular_pheno_dis[go_col].apply(lambda x: pheno_map[x])
    molecular_pheno_dis[dis_col] = molecular_pheno_dis[dis_col].apply(lambda x: dis_map[x])

    data = Data()
    data.num_nodes_dict = {
        'phenotype': len(pheno_map),
        'disease': len(dis_map)
    }
    data.edge_index_dict = {
        ('phenotype', 'dis_associated_biological_pheno', 'disease'): torch.from_numpy(biological_pheno_dis.values.T).to(torch.long),
        ('disease', 'dis_associated_biological_pheno', 'phenotype'): torch.from_numpy(biological_pheno_dis.values.T[[1,0]]).to(torch.long),
        ('phenotype', 'dis_associated_cellular_pheno', 'disease'): torch.from_numpy(cellular_pheno_dis.values.T).to(torch.long),
        ('disease', 'dis_associated_cellular_pheno', 'phenotype'): torch.from_numpy(cellular_pheno_dis.values.T[[1,0]]).to(torch.long),
        ('phenotype', 'dis_associated_molecular_pheno', 'disease'): torch.from_numpy(molecular_pheno_dis.values.T).to(torch.long),
        ('disease', 'dis_associated_molecular_pheno', 'phenotype'): torch.from_numpy(molecular_pheno_dis.values.T[[1,0]]).to(torch.long),
    }
    data.edge_reltype = {
        (h, r, t): torch.full((edge.size(1), 1), fill_value=edge_type_map[r]) for (h, r, t), edge in data.edge_index_dict.items()
    }
    data.num_relations = len(data.edge_index_dict)

    ### save chemical/disease/rel_type mapping
    if isdir(save_path):
        pass
    else:
        makedirs(save_path)

    torch.save(data, f'{save_path}/dpheno.pt')
    torch.save(pheno_map, f'{save_path}/pheno_map')
    torch.save(dis_map, f'{save_path}/dis_map')
    torch.save(edge_type_map, f'{save_path}/rel_type_map')

    print('Phenotype-Disease graph is successfully constructed.')
    return data, save_path


def build_cgd_graph(file_path = 'processed', save_path = 'processed/cgd'):
    print('This procedure may be time-consuming ...')
    print('>>> Processing Chemical-Gene-Disease Data ...')
    print('----------------------------------------------------------------------------')
    
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
    rel_type_list = list(itertools.chain(*rel_type_list))
    edge_type_map = {r: i for i, (h, r, t) in enumerate(rel_type_list)}
    
    # data construction
    data = Data()
    data.num_nodes_dict = {
        'chemical': len(chem_map),
        'gene': len(gene_map),
        'disease': len(dis_map)
    }
    
    edge_index_list = [list(data.edge_index_dict.items()) for data in data_list]
    edge_index_list = list(itertools.chain(*edge_index_list))
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
    if isdir(save_path):
        pass
    else:
        makedirs(save_path)
    
    torch.save(data, f'{save_path}/cgd.pt')
    torch.save(chem_map, f'{save_path}/chem_map')
    torch.save(gene_map, f'{save_path}/gene_map')
    torch.save(dis_map, f'{save_path}/dis_map')
    torch.save(edge_type_map, f'{save_path}/rel_type_map')
    
    print('Chemical-Gene-Disease graph is successfully constructed.')
    
    return data, save_path


def build_cgpd_graph(file_path = 'processed', save_path = 'processed/cgpd'):
    print('>>> Processing Chemical-Gene-Phenotype-Disease Data ...')
    print('----------------------------------------------------------------------------')
    
    # needs: cg, cpheno, cd, gpheno, gd
    
    # chemical
    cg_data = torch.load(f'{file_path}/cg/cg.pt')
    cg_chem_map = torch.load(f'{file_path}/cg/chem_map')
    cg_gene_map = torch.load(f'{file_path}/cg/gene_map')
    
    cpheno_data = torch.load(f'{file_path}/cpheno/cpheno.pt')
    cpheno_chem_map = torch.load(f'{file_path}/cpheno/chem_map')
    cpheno_pheno_map = torch.load(f'{file_path}/cpheno/pheno_map')
    
    cd_data = torch.load(f'{file_path}/cd/cd.pt')
    cd_chem_map = torch.load(f'{file_path}/cd/chem_map')
    cd_dis_map = torch.load(f'{file_path}/cd/dis_map')
    
    # gene
    gpheno_data = torch.load(f'{file_path}/gpheno/gpheno.pt')
    gpheno_gene_map = torch.load(f'{file_path}/gpheno/gene_map')
    gpheno_pheno_map = torch.load(f'{file_path}/gpheno/pheno_map')
    
    gd_data = torch.load(f'{file_path}/gd/gd.pt')
    gd_gene_map = torch.load(f'{file_path}/gd/gene_map')
    gd_dis_map = torch.load(f'{file_path}/gd/dis_map')
    
    # mapping
    uniq_chem = set(list(cg_chem_map.keys()) + list(cpheno_chem_map.keys()) + list(cd_chem_map.keys()))
    chem_map = {name: i for i, name in enumerate(uniq_chem)}
    
    uniq_gene = set(list(cg_gene_map.keys()) + list(gpheno_gene_map.keys()) + list(gd_gene_map.keys()))
    gene_map = {name: i for i, name in enumerate(uniq_gene)}
    
    uniq_pheno = set(list(cpheno_pheno_map.keys()) + list(gpheno_pheno_map.keys()))
    pheno_map = {name: i for i, name in enumerate(uniq_pheno)}
    
    uniq_dis = set(list(cd_dis_map.keys()) + list(gd_dis_map.keys()))
    dis_map = {name: i for i, name in enumerate(uniq_dis)}
    
    data_list = [cg_data, cpheno_data, cd_data, gpheno_data, gd_data]
    rel_type_list = [list(data.edge_index_dict.keys()) for data in data_list]
    rel_type_list = list(itertools.chain(*rel_type_list))
    edge_type_map = {r: i for i, (h, r, t) in enumerate(rel_type_list)}
    
    #
    data = Data()
    data.num_nodes_dict = {
        'chemical': len(chem_map),
        'gene': len(gene_map),
        'phenotype': len(pheno_map),
        'disease': len(dis_map)
    }
    
    edge_index_list = [list(data.edge_index_dict.items()) for data in data_list]
    edge_index_list = list(itertools.chain(*edge_index_list))
    edge_index_dict = {rel: idx for (rel, idx) in edge_index_list}
    
    print('Converting edge index')
    for (h, r, t) in tqdm(edge_index_dict.keys()):
        if (h=='chemical') & (t=='gene'): 
            old_hmap = cg_chem_map; old_tmap = cg_gene_map
            new_hmap = chem_map; new_tmap = gene_map
        elif (h=='chemical') & (t=='phenotype'): 
            old_hmap = cpheno_chem_map; old_tmap=cpheno_pheno_map
            new_hmap = chem_map; new_tmap = pheno_map
        elif (h=='chemical') & (t=='disease'): 
            old_hmap = cd_chem_map; old_tmap = cd_dis_map
            new_hmap = chem_map; new_tmap = dis_map
        elif (h=='gene') & (t=='phenotype'): 
            old_hmap = gpheno_gene_map;   old_tmap=gpheno_pheno_map
            new_hmap = gene_map; new_tmap = pheno_map
        elif (h=='phenotype') & (t=='gene'): 
            old_hmap = gpheno_pheno_map ;  old_tmap=gpheno_gene_map
            new_hmap = pheno_map; new_tmap = gene_map
        elif (h=='gene') & (t=='disease'): 
            old_hmap = gd_gene_map; old_tmap = gd_dis_map
            new_hmap = gene_map; new_tmap = dis_map
        
        heads, tails = edge_index_dict[(h, r, t)].numpy()
        # value to key
        heads_id = np.array(list(old_hmap.keys()))[heads]
        tails_id = np.array(list(old_tmap.keys()))[tails]
        
        # key to new value
        new_heads = np.array(list(map(lambda x: new_hmap[x], heads_id)))
        new_tails = np.array(list(map(lambda x: new_tmap[x], tails_id)))
        
        edge_index_dict[(h, r, t)] = torch.from_numpy(np.stack([new_heads, new_tails])).to(torch.long)
        
    data.edge_index_dict = edge_index_dict
    data.edge_reltype = {
        (h, r, t): torch.full((edge.size(1), 1), fill_value=edge_type_map[r]) for (h, r, t), edge in data.edge_index_dict.items()
    }
    data.num_relations = len(edge_type_map)

    ### save chemical/disease/rel_type mapping
    if isdir(save_path):
        pass
    else:
        makedirs(save_path)

    torch.save(data, f'{save_path}/cgpd.pt')
    torch.save(chem_map, f'{save_path}/chem_map')
    torch.save(gene_map, f'{save_path}/gene_map')
    torch.save(pheno_map, f'{save_path}/pheno_map')
    torch.save(dis_map, f'{save_path}/dis_map')
    torch.save(edge_type_map, f'{save_path}/rel_type_map')
    
    print('Chemical-Gene-Phenotype-Disease graph is successfully constructed.')
    
    return data, save_path


def build_ctd_graph(file_path = 'processed', save_path = 'processed/ctd'):
    print('>>> Processing Comparative Toxicogenomics Data ...')
    print('----------------------------------------------------------------------------')
    
    # cgpd, cpath, cgo
    # gpath, ggo
    # dpath, dpheno
    
    # chemical
    cgpd_data = torch.load(f'{file_path}/cgpd/cgpd.pt')
    cgpd_chem_map = torch.load(f'{file_path}/cgpd/chem_map')
    cgpd_gene_map = torch.load(f'{file_path}/cgpd/gene_map')
    cgpd_pheno_map = torch.load(f'{file_path}/cgpd/pheno_map')
    cgpd_dis_map = torch.load(f'{file_path}/cgpd/dis_map')
    
    cpath_data = torch.load(f'{file_path}/cpath/cpath.pt')
    cpath_chem_map = torch.load(f'{file_path}/cpath/chem_map')
    cpath_path_map = torch.load(f'{file_path}/cpath/path_map')
    
    cgo_data = torch.load(f'{file_path}/cgo/cgo.pt')
    cgo_chem_map = torch.load(f'{file_path}/cgo/chem_map')
    cgo_go_map = torch.load(f'{file_path}/cgo/go_map')
    
    # gene
    gpath_data = torch.load(f'{file_path}/gpath/gpath.pt')
    gpath_gene_map = torch.load(f'{file_path}/gpath/gene_map')
    gpath_path_map = torch.load(f'{file_path}/gpath/path_map')
    
    ggo_data = torch.load(f'{file_path}/ggo/ggo.pt')
    ggo_gene_map = torch.load(f'{file_path}/ggo/gene_map')
    ggo_go_map = torch.load(f'{file_path}/ggo/go_map')
    
    # disease
    dpath_data = torch.load(f'{file_path}/dpath/dpath.pt')
    dpath_dis_map = torch.load(f'{file_path}/dpath/dis_map')
    dpath_path_map = torch.load(f'{file_path}/dpath/path_map')
    
    dpheno_data = torch.load(f'{file_path}/dpheno/dpheno.pt')
    dpheno_dis_map = torch.load(f'{file_path}/dpheno/dis_map')
    dpheno_pheno_map = torch.load(f'{file_path}/dpheno/pheno_map')
    
    # mapping
    uniq_chem = set(list(cgpd_chem_map.keys()) + list(cpath_chem_map.keys()) + list(cgo_chem_map.keys()))
    chem_map = {name: i for i, name in enumerate(uniq_chem)}
    
    uniq_gene = set(list(cgpd_gene_map.keys()) + list(gpath_gene_map.keys()) + list(ggo_gene_map.keys()))
    gene_map = {name: i for i, name in enumerate(uniq_gene)}
    
    uniq_dis = set(list(cgpd_dis_map.keys()) + list(dpath_dis_map.keys()) + list(dpheno_dis_map.keys()))
    dis_map = {name: i for i, name in enumerate(uniq_dis)}
    
    uniq_path = set(list(cpath_path_map.keys()) + list(gpath_path_map.keys()) + list(dpath_path_map.keys()))
    path_map = {name: i for i, name in enumerate(uniq_path)}
    
    uniq_go = set(list(cgo_go_map.keys()) + list(ggo_go_map.keys()))
    go_map = {name: i for i, name in enumerate(uniq_go)}
    
    uniq_pheno = set(list(cgpd_pheno_map.keys()) + list(dpheno_pheno_map.keys()))
    pheno_map = {name: i for i, name in enumerate(uniq_pheno)}    
    
    data_list = [cgpd_data, cpath_data, cgo_data, gpath_data, ggo_data, dpath_data, dpheno_data]
    rel_type_list = [list(data.edge_index_dict.keys()) for data in data_list]
    rel_type_list = list(itertools.chain(*rel_type_list))
    edge_type_map = {r: i for i, (h, r, t) in enumerate(rel_type_list)}
    
    #
    data = Data()
    data.num_nodes_dict = {
        'chemical': len(chem_map),
        'gene': len(gene_map),
        'phenotype': len(pheno_map),
        'disease': len(dis_map),
        'pathway': len(path_map),
        'gene_ontology': len(go_map)
    }
    
    edge_index_list = [list(data.edge_index_dict.items()) for data in data_list]
    edge_index_list = list(itertools.chain(*edge_index_list))
    edge_index_dict = {rel: idx for (rel, idx) in edge_index_list}
    
    print('Converting edge index')
    for (h, r, t) in tqdm(edge_index_dict.keys()):
        if (h=='chemical') & (t=='gene'): 
            old_hmap = cgpd_chem_map; old_tmap = cgpd_gene_map
            new_hmap = chem_map; new_tmap = gene_map
        elif (h=='chemical') & (t=='phenotype'): 
            old_hmap = cgpd_chem_map; old_tmap = cgpd_pheno_map
            new_hmap = chem_map; new_tmap = pheno_map
        elif (h=='chemical') & (t=='disease'): 
            old_hmap = cgpd_chem_map; old_tmap = cgpd_dis_map
            new_hmap = chem_map; new_tmap = dis_map
        elif (h=='chemical') & (t=='pathway'):
            old_hmap = cpath_chem_map; old_tmap = cpath_path_map
            new_hmap = chem_map; new_tmap = path_map
        elif (h=='chemical') & (t=='gene_ontology'):
            old_hmap = cgo_chem_map; old_tmap = cgo_go_map
            new_hmap = chem_map; new_tmap = go_map
        elif (h=='gene') & (t=='phenotype'): 
            old_hmap = cgpd_gene_map; old_tmap = cgpd_pheno_map
            new_hmap = gene_map; new_tmap = pheno_map
        elif (h=='phenotype') & (t=='gene'): 
            old_hmap = cgpd_pheno_map ;  old_tmap = cgpd_gene_map
            new_hmap = pheno_map; new_tmap = gene_map
        elif (h=='gene') & (t=='disease'): 
            old_hmap = cgpd_gene_map; old_tmap = cgpd_dis_map
            new_hmap = gene_map; new_tmap = dis_map
        elif (h=='gene') & (t=='pathway'):
            old_hmap = gpath_gene_map; old_tmap = gpath_path_map
            new_hmap = gene_map; new_tmap = path_map
        elif (h=='pathway') & (t=='gene'):
            old_hmap = gpath_path_map; old_tmap = gpath_gene_map
            new_hmap = path_map; new_tmap = gene_map
        elif (h=='gene') & (t=='gene_ontology'):
            old_hmap = ggo_gene_map; old_tmap = ggo_go_map
            new_hmap = gene_map; new_tmap = go_map
        elif (h=='gene_ontology') & (t=='gene'):
            old_hmap = ggo_go_map; old_tmap = ggo_gene_map
            new_hmap = go_map; new_tmap = gene_map
        elif (h=='disease') & (t=='pathway'):
            old_hmap = dpath_dis_map; old_tmap = dpath_path_map
            new_hmap = dis_map; new_tmap = path_map
        elif (h=='pathway') & (t=='disease'):
            old_hmap = dpath_path_map; old_tmap = dpath_dis_map
            new_hmap = path_map; new_tmap = dis_map
        elif (h=='disease') & (t=='phenotype'):
            old_hmap = dpheno_dis_map; old_tmap = dpheno_pheno_map
            new_hmap = dis_map; new_tmap = pheno_map
        elif (h=='phenotype') & (t=='disease'):
            old_hmap=dpheno_pheno_map; old_tmap=dpheno_dis_map
            new_hmap=pheno_map; new_tmap=dis_map
        
        heads, tails = edge_index_dict[(h, r, t)].numpy()
        # value to key
        heads_id = np.array(list(old_hmap.keys()))[heads]
        tails_id = np.array(list(old_tmap.keys()))[tails]
        
        # key to new value
        new_heads = np.array(list(map(lambda x: new_hmap[x], heads_id)))
        new_tails = np.array(list(map(lambda x: new_tmap[x], tails_id)))
        
        edge_index_dict[(h, r, t)] = torch.from_numpy(np.stack([new_heads, new_tails])).to(torch.long)
        
    data.edge_index_dict = edge_index_dict
    data.edge_reltype = {
        (h, r, t): torch.full((edge.size(1), 1), fill_value=edge_type_map[r]) for (h, r, t), edge in data.edge_index_dict.items()
    }
    data.num_relations = len(edge_type_map)

    ### save chemical/disease/rel_type mapping
    if isdir(save_path):
        pass
    else:
        makedirs(save_path)

    torch.save(data, f'{save_path}/ctd.pt')
    torch.save(chem_map, f'{save_path}/chem_map')
    torch.save(gene_map, f'{save_path}/gene_map')
    torch.save(pheno_map, f'{save_path}/pheno_map')
    torch.save(dis_map, f'{save_path}/dis_map')
    torch.save(path_map, f'{save_path}/path_map')
    torch.save(go_map, f'{save_path}/go_map')
    torch.save(edge_type_map, f'{save_path}/rel_type_map')
    
    print('CTD graph is successfully constructed.')
    
    return data, save_path


def build_benchmarks(data_type, train_frac, valid_frac):
    print('>>> Build Benchmark Dataset ...')
    
    ### create data
    if data_type == 'cd':
        data, save_path = build_cd_graph()
    elif data_type == 'cg':
        data, save_path = build_cg_graph()
    elif data_type == 'cpath':
        data, save_path = build_cpath_graph()
    elif data_type == 'cpheno':
        data, save_path = build_cpheno_graph()
    elif data_type == 'cgo':
        data, save_path = build_cgo_graph()
    elif data_type == 'gpath':
        data, save_path = build_gpath_graph()
    elif data_type == 'gpheno':
        data, save_path = build_gpheno_graph()
    elif data_type == 'ggo':
        data, save_path = build_ggo_graph()
    elif data_type == 'gd':
        data, save_path = build_gd_graph()
    elif data_type == 'dpath':
        data, save_path = build_dpath_graph()
    elif data_type == 'dpheno':
        data, save_path = build_dpheno_graph()
    elif data_type == 'cgd':
        data, save_path = build_cgd_graph()
    elif data_type == 'cgpd':
        data, save_path = build_cgpd_graph()
    elif data_type == 'ctd':
        data, save_path = build_ctd_graph()
    
    ### split data
    train_data, valid_data, test_data = split_data(data, train_frac, valid_frac)
    
    ### negative sampling
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
    
    print('Negative sampling for validation data')
    valid_data = negative_sampling(valid_data, all_true_head, all_true_tail)
    print('Negative sampling for test data')
    test_data = negative_sampling(test_data, all_true_head, all_true_tail)

    ### save splitted data
    torch.save(train_data, f'{save_path}/train_{data_type}.pt')
    torch.save(valid_data, f'{save_path}/valid_{data_type}.pt')
    torch.save(test_data, f'{save_path}/test_{data_type}.pt')
    
    print('Graph construction is completed!')


if __name__ == '__main__':
    build_cg_graph()
    build_cpheno_graph()
    build_cpath_graph()
    build_cgo_graph()
    build_gd_graph()
    build_gpheno_graph()
    build_gpath_graph()
    build_ggo_graph()
    build_dpheno_graph()
    build_dpath_graph()
    build_benchmarks('cd', 0.9, 0.05)
    # build_benchmarks('cg', 0.9, 0.05)
    # build_benchmarks('gd', 0.98, 0.01)
    build_benchmarks('cgd', 0.98, 0.01)
    build_benchmarks('cgpd', 0.98, 0.01)
    build_benchmarks('ctd', 0.98, 0.01)
