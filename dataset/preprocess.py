from os import mkdir
from os.path import isdir, isfile

import random
import itertools
from itertools import repeat
from collections import defaultdict

import pandas as pd
from tqdm import tqdm

import torch
from torch_geometric.data import Data

from generate_negative import NegativeSampling
from data_download import download_ctd_data


chem_col = 'ChemicalID'
dis_col = 'DiseaseID'
gene_col = 'GeneID'
path_col = 'PathwayID'


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


def negative_sampling(data):
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

    true_head, true_tail = defaultdict(list), defaultdict(list)
    for i in tqdm(range(len(triples['head']))):
        head, relation, tail = triples['head'][i].item(), triples['relation'][i].item(), triples['tail'][i].item()
        head_type, tail_type = triples['head_type'][i], triples['tail_type'][i]
        true_head[(relation, tail)].append(head)
        true_tail[(head, relation)].append(tail)

    head_negative_sampling = NegativeSampling(triples, 500, 'head-batch', true_head, true_tail, data.num_nodes_dict)
    head_negative = []
    for i in tqdm(range(len(triples['head']))):
        head_negative.append(head_negative_sampling.__getitem__(i)[1])

    tail_negative_sampling = NegativeSampling(triples, 500, 'tail-batch', true_head, true_tail, data.num_nodes_dict)
    tail_negative = []
    for i in tqdm(range(len(triples['tail']))):
        tail_negative.append(tail_negative_sampling.__getitem__(i)[1])

    head_negative = torch.stack(head_negative)
    tail_negative = torch.stack(tail_negative)
    
    data['head_neg'] = {key: head_negative[:v.shape[1]] for key, v in data.edge_index_dict.items()}
    data['tail_neg'] = {key: tail_negative[:v.shape[1]] for key, v in data.edge_index_dict.items()}
    
    return data


def build_cd_graph(file_path = 'raw', save_path = 'processed/cd'):
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
        raise ValueError(f'duplicated direct evidence: {dir_dup_num}')
    else: 
        print(f'Number of Duplicated DirectEvidence: {dir_dup_num}')
    
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
        rel: torch.full((edge.size(1), 1), fill_value = i).to(torch.long) for i, (rel, edge) in enumerate(data.edge_index_dict.items())
    }
    data.num_relations = len(data.edge_index_dict)
    
    ### save chemical/disease/rel_type mapping
    if isdir(save_path):
        pass
    else:
        mkdir(save_path)
    
    torch.save(data, f'{save_path}/cd.pt')
    torch.save(chem_map, f'{save_path}/chem_map')
    torch.save(dis_map, f'{save_path}/dis_map')
    torch.save(edge_type_map, f'{save_path}/rel_type_map')
    
    print('Chemical-Disease graph is successfully constructed.')
    
    return data, save_path


def build_cg_graph(type, file_path = 'raw', save_path = 'processed/cg'):
    '''
    Build a Chemical-Gene interaction graph
    Parameter
    ----------
        type: str (v1 or v2)
            v1: simple c-g
            v2: gene entity to multiple gene/mRNA/etc entity
    ----------
    '''
    print(f'>>> Processing Chemical-Gene Data {type} ...')
    print('----------------------------------------------------------------------------')
    
    chem_gene_tmp = pd.read_csv(f'{file_path}/CTD_chem_gene_ixns.csv.gz', skiprows = list(range(27)) + [28], compression = 'gzip')

    # delete data which are not specified the organism
    org_na_idx = chem_gene_tmp['Organism'].isna()
    geneform_na_idx = chem_gene_tmp['GeneForms'].isna()
    chem_gene = chem_gene_tmp[~(org_na_idx|geneform_na_idx)]

    if type == 'v1':
        # remove duplicated chem-gene pair
        chem_gene = chem_gene[[chem_col, gene_col]].drop_duplicates()

        uniq_chem = chem_gene[chem_col].unique()
        chem_map = {name: i for i, name in enumerate(uniq_chem)}

        uniq_gene = chem_gene[gene_col].unique()
        gene_map = {name: i for i, name in enumerate(uniq_gene)}

        edge_type_map = {
            'chem_inferred_gene': 0
        }

        # mapping the chemical and disease id
        inferred_chem_gene = chem_gene.copy()
        inferred_chem_gene[chem_col] = inferred_chem_gene[chem_col].apply(lambda x: chem_map[x])
        inferred_chem_gene[gene_col] = inferred_chem_gene[gene_col].apply(lambda x: gene_map[x])

        data = Data()
        data.num_nodes_dict = {
            'chemical': len(chem_map),
            'gene': len(gene_map)
        }
        data.edge_index_dict = {
            ('chemical', 'chem_inferred_gene', 'gene'): torch.from_numpy(inferred_chem_gene.values.T).to(torch.long)
        }
        data.edge_reltype = {
            rel: torch.full((edge.size(1), 1), fill_value = i).to(torch.long) for i, (rel, edge) in enumerate(data.edge_index_dict.items())
        }
        data.num_relations = len(data.edge_index_dict)

        ### save mapping
        save_path = save_path + '/v1'
        if isdir(save_path):
            pass
        else:
            mkdir(save_path)
        
        torch.save(data, f'{save_path}/cg.pt')
        torch.save(chem_map, f'{save_path}/chem_map')
        torch.save(gene_map, f'{save_path}/dis_map')
        torch.save(edge_type_map, f'{save_path}/rel_type_map')
    
    elif type == 'v2':
        split_geneform = chem_gene['GeneForms'].map(lambda x: x.split('|'))
        uniq_geneform = set(itertools.chain(*split_geneform))

        num_geneforms = split_geneform.map(lambda x: len(x))

        single_chem_gene = chem_gene[num_geneforms == 1]
        double_chem_gene = chem_gene[num_geneforms == 2]
        triple_chem_gene = chem_gene[num_geneforms == 3]

        ### split dataframe per geneform
        single_chem_gene['GeneForms'].value_counts()
        geneform_df_dict = {g: chem_gene[chem_gene['GeneForms']==g] for g in uniq_geneform}

        ### insert data which have multiple geneforms to geneform_df_dict
        double_chem_gene = double_chem_gene.drop_duplicates([chem_col, gene_col, 'GeneForms'])
        
        split_double_geneform1 = split_geneform[num_geneforms == 2].map(lambda x: x[0])
        split_double_geneform2 = split_geneform[num_geneforms == 2].map(lambda x: x[1])
        
        double_chem_gene['GeneForms'] = split_double_geneform1
        geneform_df_dict = {g: pd.concat([geneform_df_dict[g], double_chem_gene[double_chem_gene['GeneForms'] == g]]) for g in uniq_geneform}
        
        double_chem_gene['GeneForms'] = split_double_geneform2
        geneform_df_dict = {g: pd.concat([geneform_df_dict[g], double_chem_gene[double_chem_gene['GeneForms'] == g]]) for g in uniq_geneform}
        
        ### insert data which have triple geneforms to geneform_df_dict
        triple_chem_gene = triple_chem_gene.drop_duplicates([chem_col, gene_col, 'GeneForms'])
        
        split_triple_geneform1 = split_geneform[num_geneforms == 3].map(lambda x: x[0])
        split_triple_geneform2 = split_geneform[num_geneforms == 3].map(lambda x: x[1])
        split_triple_geneform3 = split_geneform[num_geneforms == 3].map(lambda x: x[2])
        
        triple_chem_gene['GeneForms'] = split_triple_geneform1
        geneform_df_dict = {g: pd.concat([geneform_df_dict[g], triple_chem_gene[triple_chem_gene['GeneForms'] == g]]) for g in uniq_geneform}
        triple_chem_gene['GeneForms'] = split_triple_geneform2
        geneform_df_dict = {g: pd.concat([geneform_df_dict[g], triple_chem_gene[triple_chem_gene['GeneForms'] == g]]) for g in uniq_geneform}
        triple_chem_gene['GeneForms'] = split_triple_geneform3
        geneform_df_dict = {g: pd.concat([geneform_df_dict[g], triple_chem_gene[triple_chem_gene['GeneForms'] == g]]) for g in uniq_geneform}
        
        geneform_df_dict = {k: v.drop_duplicates([chem_col, gene_col])[[chem_col, gene_col]] for k, v in geneform_df_dict.items()}
        
        ### build graph
        uniq_chem = chem_gene[chem_col].unique()
        chem_map = {name: i for i, name in enumerate(uniq_chem)}
        
        uniq_gene = {k: v[gene_col].unique() for k, v in geneform_df_dict.items()}
        gene_map = {k: {name: i for i, name in enumerate(v)} for k, v in uniq_gene.items()}
        
        geneform_edge_index = {k: pd.DataFrame([v[chem_col].map(lambda x: chem_map[x]),
                                               v[gene_col].map(lambda x: gene_map[k][x])]) for k, v in geneform_df_dict.items()}
        edge_type_map = {('chemical', f'chem_inferred_{k.replace(" " , "")}', k): i for i, k in enumerate(geneform_edge_index)}
        
        data = Data()
        data.num_nodes_dict = {
            'chemical': len(chem_map)
        }
        data.num_nodes_dict.update({k: len(v) for k, v in gene_map.items()})
        data.edge_index_dict = {
            k: torch.from_numpy(geneform_edge_index[k[-1]].values).to(torch.long) for k in edge_type_map.keys()
        }
        data.edge_reltype = {
            rel: torch.full((edge.size(1), 1), fill_value = edge_type_map[rel]).to(torch.long) for i, (rel, edge) in enumerate(data.edge_index_dict.items())
        }
        data.num_relations = len(data.edge_index_dict)
        
        ### save mapping
        save_path = save_path + '/v2'
        if isdir(save_path):
            pass
        else:
            mkdir(save_path)
        
        torch.save(data, f'{save_path}/cg.pt')
        torch.save(chem_map, f'{save_path}/chem_map')
        torch.save(gene_map, f'{save_path}/dis_map')
        torch.save(edge_type_map, f'{save_path}/rel_type_map')

    print('Chemical-Gene Graph is successfully constructed.')
    
    return data, save_path


def build_cpath_graph(file_path = 'raw', save_path = 'processed/cpath'):
    print('>>> Processing Chemical-Pathway Data ...')
    print('----------------------------------------------------------------------------')
    
    # this file does not have duplicated (chem, pathway) pair
    chem_path_tmp = pd.read_csv(f'{file_path}/CTD_chem_pathways_enriched.csv.gz', skiprows = list(range(27))+[28], compression = 'gzip')
    
    ### build graph
    uniq_chem = chem_path_tmp[chem_col].unique()
    chem_map = {name: i for i, name in enumerate(uniq_chem)}

    uniq_path = chem_path_tmp[path_col].unique()
    path_map = {name: i for i, name in enumerate(uniq_path)}

    edge_type_map = {
        'chem_inferred_path': 0
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
        ('chemical', 'chem_inferred_path', 'pathway'): torch.from_numpy(chem_path.values.T).to(torch.long),
    }
    data.edge_reltype = {
        rel: torch.full((edge.size(1), 1), fill_value = i).to(torch.long) for i, (rel, edge) in enumerate(data.edge_index_dict.items())
    }
    data.num_relations = len(data.edge_index_dict)
    
    ### save mapping
    if isdir(save_path):
        pass
    else:
        mkdir(save_path)
    
    torch.save(data, f'{save_path}/cpath.pt')
    torch.save(chem_map, f'{save_path}/chem_map')
    torch.save(path_map, f'{save_path}/path_map')
    torch.save(edge_type_map, f'{save_path}/rel_type_map')
    
    print('Chemical-Pathway graph is successfully constructed.')
    
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
        'gene_related_path': 0
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
        ('gene', 'gene_related_path', 'pathway'): torch.from_numpy(gene_path.values.T).to(torch.long),
    }
    data.edge_reltype = {
        rel: torch.full((edge.size(1), 1), fill_value = i).to(torch.long) for i, (rel, edge) in enumerate(data.edge_index_dict.items())
    }
    data.num_relations = len(data.edge_index_dict)
    
    ### save mapping
    if isdir(save_path):
        pass
    else:
        mkdir(save_path)
    
    torch.save(data, f'{save_path}/cpath.pt')
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
        raise ValueError(f'duplicated direct evidence: {dir_dup_num}')
    else: 
        print(f'Number of Duplicated DirectEvidence: {dir_dup_num}')
    
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
        rel: torch.full((edge.size(1), 1), fill_value = i).to(torch.long) for i, (rel, edge) in enumerate(data.edge_index_dict.items())
    }
    data.num_relations = len(data.edge_index_dict)
    
    ### save chemical/disease/rel_type mapping
    if isdir(save_path):
        pass
    else:
        mkdir(save_path)
    
    torch.save(data, f'{save_path}/gd.pt')
    torch.save(gene_map, f'{save_path}/gene_map')
    torch.save(dis_map, f'{save_path}/dis_map')
    torch.save(edge_type_map, f'{save_path}/rel_type_map')
    
    print('Gene-Disease graph is successfully constructed.')
    
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
        'disease_related_path': 0
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
        ('disease', 'disease_related_path', 'pathway'): torch.from_numpy(dis_path.values.T).to(torch.long),
    }
    data.edge_reltype = {
        rel: torch.full((edge.size(1), 1), fill_value = i).to(torch.long) for i, (rel, edge) in enumerate(data.edge_index_dict.items())
    }
    data.num_relations = len(data.edge_index_dict)
    
    ### save mapping
    if isdir(save_path):
        pass
    else:
        mkdir(save_path)

    torch.save(data, f'{save_path}/dpath.pt')
    torch.save(dis_map, f'{save_path}/dis_map')
    torch.save(path_map, f'{save_path}/path_map')
    torch.save(edge_type_map, f'{save_path}/rel_type_map')
    
    print('Disease-Pathway graph is successfully constructed.')
    
    return data, save_path


def build_cgd_graph(file_path = 'raw', save_path = 'processed/cgd'):
    print('This procedure may be time-consuming ...')
    print('>>> Processing Chemical-Gene-Disease Data ...')
    print('----------------------------------------------------------------------------')
    
    chem_dis_tmp = pd.read_csv(f'{file_path}/CTD_chemicals_diseases.csv.gz', skiprows = list(range(27))+[28], compression = 'gzip')
    chem_gene_tmp = pd.read_csv(f'{file_path}/CTD_chem_gene_ixns.csv.gz', skiprows = list(range(27))+[28], compression = 'gzip')
    gene_dis_tmp = pd.read_csv(f'{file_path}/CTD_genes_diseases.csv.gz', skiprows = list(range(27))+[28], compression = 'gzip')

    ### delete data which have 'therapeutic' DirectEvidence
    cd_thera_idx = chem_dis_tmp.DirectEvidence == 'therapeutic'
    chem_dis = chem_dis_tmp[~cd_thera_idx]
    del chem_dis_tmp
    
    gd_thera_idx = gene_dis_tmp.DirectEvidence == 'therapeutic'
    gene_dis = gene_dis_tmp[~gd_thera_idx]
    del gene_dis_tmp
    
    # delete data which are not specified the organism
    org_na_idx = chem_gene_tmp['Organism'].isna()
    geneform_na_idx = chem_gene_tmp['GeneForms'].isna()
    chem_gene = chem_gene_tmp[~(org_na_idx|geneform_na_idx)]
    del chem_gene_tmp
    
    ### curated (DirectEvidence: marker/mechanism) edge index
    # chemical-disease
    curated_chem_dis = chem_dis[~chem_dis.DirectEvidence.isna()][[chem_col, dis_col]]
    dir_cd_dup_num = curated_chem_dis.duplicated(keep = False).sum()
    if dir_cd_dup_num != 0:
        raise ValueError(f'duplicated direct evidence of chem-dis: {dir_cd_dup_num}')
    else: 
        print(f'Number of Duplicated DirectEvidence of Chem-Dis: {dir_cd_dup_num}')
    
    # gene-disease
    curated_gene_dis = gene_dis[~gene_dis.DirectEvidence.isna()][[gene_col, dis_col]]
    dir_gd_dup_num = curated_gene_dis.duplicated(keep = False).sum()
    if dir_gd_dup_num != 0:
        raise ValueError(f'duplicated direct evidence of gene-dis: {dir_gd_dup_num}')
    else: 
        print(f'Number of Duplicated DirectEvidence of Gene-Dis: {dir_gd_dup_num}')
    
    ### inferred edge index
    # (c, d) pairs which have DirectEvidence and Inferred Relation
    dup_chem_dis_idx = chem_dis[[chem_col, dis_col]].duplicated(keep = False)
    dup_chem_dis = chem_dis[dup_chem_dis_idx]
    dup_dir_chem_dis = dup_chem_dis[~dup_chem_dis.DirectEvidence.isna()][[chem_col, dis_col]]
    # (c, d) pairs which have Inferred Relation and drops duplicate
    inferred_chem_dis = chem_dis[chem_dis.DirectEvidence.isna()][[chem_col, dis_col]]
    inferred_chem_dis = inferred_chem_dis.drop_duplicates()
    # merge dup_dir_chem_dis and drop which duplicated
    inferred_chem_dis = pd.concat([dup_dir_chem_dis, inferred_chem_dis])
    inferred_chem_dis = inferred_chem_dis.drop_duplicates(keep = False)
    
    # (g, d) pairs which have DirecEvidence and Inferred Relation
    dup_gene_dis_idx = gene_dis[[gene_col, dis_col]].duplicated(keep = False)
    dup_gene_dis = gene_dis[dup_gene_dis_idx]
    dup_dir_gene_dis = dup_gene_dis[~dup_gene_dis.DirectEvidence.isna()][[gene_col, dis_col]]
    # (g, d) pairs which have Inferred Relation and drops duplicate
    inferred_gene_dis = gene_dis[gene_dis.DirectEvidence.isna()][[gene_col, dis_col]]
    inferred_gene_dis = inferred_gene_dis.drop_duplicates()
    # merge dup_dir_chem_dis and drop which duplicated
    inferred_gene_dis = pd.concat([dup_dir_gene_dis, inferred_gene_dis])
    inferred_gene_dis = inferred_gene_dis.drop_duplicates(keep = False)
    
    # (c, g)
    inferred_chem_gene = chem_gene[[chem_col, gene_col]].drop_duplicates()
    
    ### build graph
    # mapping of unique chemical, disease, and gene
    uniq_chem = pd.concat([chem_dis[chem_col], chem_gene[chem_col]]).unique()
    chem_map = {name: i for i, name in enumerate(uniq_chem)}

    uniq_dis = pd.concat([chem_dis[dis_col], gene_dis[dis_col]]).unique()
    dis_map = {name: i for i, name in enumerate(uniq_dis)}
    
    uniq_gene = pd.concat([chem_gene[gene_col], gene_dis[gene_col]]).unique()
    gene_map = {name: i for i, name in enumerate(uniq_gene)}

    del chem_dis
    del chem_gene
    del gene_dis

    edge_type_map = {
        'chem_curated_dis': 0,
        'chem_inferred_dis': 1,
        'gene_curated_dis': 2,
        'gene_inferred_dis': 3,
        'chem_inferred_gene': 4
    }

    # mapping the chemical and disease id
    curated_chem_dis[chem_col] = curated_chem_dis[chem_col].apply(lambda x: chem_map[x])
    curated_chem_dis[dis_col] = curated_chem_dis[dis_col].apply(lambda x: dis_map[x])

    inferred_chem_dis[chem_col] = inferred_chem_dis[chem_col].apply(lambda x: chem_map[x])
    inferred_chem_dis[dis_col] = inferred_chem_dis[dis_col].apply(lambda x: dis_map[x])
    
    curated_gene_dis[gene_col] = curated_gene_dis[gene_col].apply(lambda x: gene_map[x])
    curated_gene_dis[dis_col] = curated_gene_dis[dis_col].apply(lambda x: dis_map[x])

    inferred_gene_dis[gene_col] = inferred_gene_dis[gene_col].apply(lambda x: gene_map[x])
    inferred_gene_dis[dis_col] = inferred_gene_dis[dis_col].apply(lambda x: dis_map[x])

    inferred_chem_gene[chem_col] = inferred_chem_gene[chem_col].apply(lambda x: chem_map[x])
    inferred_chem_gene[gene_col] = inferred_chem_gene[gene_col].apply(lambda x: gene_map[x])
    

    data = Data()
    data.num_nodes_dict = {
        'chemical': len(chem_map),
        'gene': len(gene_map),
        'disease': len(dis_map)
    }
    data.edge_index_dict = {
        ('chemical', 'chem_curated_dis', 'disease'): torch.from_numpy(curated_chem_dis.values.T).to(torch.long),
        ('chemical', 'chem_inferred_dis', 'disease'): torch.from_numpy(inferred_chem_dis.values.T).to(torch.long),
        ('gene', 'gene_curated_dis', 'disease'): torch.from_numpy(curated_gene_dis.values.T).to(torch.long),
        ('gene', 'gene_inferred_dis', 'disease'): torch.from_numpy(inferred_gene_dis.values.T).to(torch.long),
        ('chem', 'chem_inferred_gene', 'gene'): torch.from_numpy(inferred_chem_gene.values.T).to(torch.long),
    }
    data.edge_reltype = {
        rel: torch.full((edge.size(1), 1), fill_value = i).to(torch.long) for i, (rel, edge) in enumerate(data.edge_index_dict.items())
    }
    data.num_relations = len(data.edge_index_dict)
    
    ### save chemical/disease/rel_type mapping
    if isdir(save_path):
        pass
    else:
        mkdir(save_path)
    
    torch.save(data, f'{save_path}/cgd.pt')
    torch.save(chem_map, f'{save_path}/chem_map')
    torch.save(gene_map, f'{save_path}/gene_map')
    torch.save(dis_map, f'{save_path}/dis_map')
    torch.save(edge_type_map, f'{save_path}/rel_type_map')
    
    print('Chemical-Gene-Disease graph is successfully constructed.')
    
    return data, save_path


def build_benchmarks(data_type, train_frac, valid_frac):
    print('>>> Build Benchmark Dataset ...')
    
    ### create data
    if data_type == 'cd':
        data, save_path = build_cd_graph()
    elif data_type == 'cg-v1':
        data, save_path = build_cg_graph('v1')
    elif data_type == 'cg-v2':
        data, save_path = build_cg_graph('v2')
    elif data_type == 'cpath':
        data, save_path = build_cpath_graph()
    elif data_type == 'gpath':
        data, save_path = build_gpath_graph()
    elif data_type == 'gd':
        data, save_path = build_gd_graph()
    elif data_type == 'dpath':
        data, save_path = build_dpath_graph()
    # elif data_type == :
    
    ### split data
    train_data, valid_data, test_data = split_data(data, train_frac, valid_frac)
    
    ### negative sampling
    print('Negative Sampling for Validation Data')
    valid_data = negative_sampling(valid_data)
    print('Negative Sampling for Test Data')
    test_data = negative_sampling(test_data)

    ### save splitted data
    torch.save(train_data, f'{save_path}/train_{data_type}.pt')
    torch.save(valid_data, f'{save_path}/valid_{data_type}.pt')
    torch.save(test_data, f'{save_path}/test_{data_type}.pt')
    
    print('Graph Construction is Completed!')


# build_benchmarks('cd', 0.9, 0.05)
# build_benchmarks('cg-v1', 0.9, 0.05)
# build_benchmarks('cg-v2', 0.9, 0.05)
# build_benchmarks('cpath', 0.9, 0.05)
# build_benchmarks('gpath', 0.9, 0.05)
# build_benchmarks('cgd', 0.98, 0.01)


