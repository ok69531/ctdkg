import os
import re
import torch
import openai
import warnings

import numpy as np
import pandas as pd

from tqdm.autonotebook import tqdm

warnings.filterwarnings('ignore')


def extract_unique_ids(edge_index, entity_type: str):
    id_list = []
    
    for (h, r, t), v in edge_index.items():
        if h == entity_type:
            id_list.append(v[0])
        elif t == entity_type:
            id_list.append(v[1])
    
    id_list = torch.unique(torch.cat(id_list)).tolist()
    
    return id_list


def load_entity_name_map(entity_type: str):
    chem_id_map = torch.load('processed/ctd/chem_map')
    gene_id_map = torch.load('processed/ctd/gene_map')
    dis_id_map = torch.load('processed/ctd/dis_map')
    pheno_id_map = torch.load('processed/ctd/pheno_map')
    path_id_map = torch.load('processed/ctd/path_map')
    go_id_map = torch.load('processed/ctd/go_map')
    
    id_maps = (chem_id_map, gene_id_map, dis_id_map, pheno_id_map, path_id_map, go_id_map)
    
    inv_chem_map = {v: k for k, v in chem_id_map.items()}
    inv_gene_map = {v: k for k, v in gene_id_map.items()}
    inv_dis_map = {v: k for k, v in dis_id_map.items()}
    inv_pheno_map = {v: k for k, v in pheno_id_map.items()}
    inv_path_map = {v: k for k, v in path_id_map.items()}
    inv_go_map = {v: k for k, v in go_id_map.items()}
    
    inv_id_maps = (inv_chem_map, inv_gene_map, inv_dis_map, inv_pheno_map, inv_path_map, inv_go_map)
    
    if entity_type == 'chemical':
        return id_maps, inv_id_maps
    elif entity_type == 'gene':
        return id_maps, inv_id_maps
    elif entity_type == 'disease':
        return id_maps, inv_id_maps
    elif entity_type == 'phenotype':
        return id_maps, inv_id_maps
    elif entity_type == 'pathway':
        return id_maps, inv_id_maps
    elif entity_type == 'go':
        return id_maps, inv_id_maps


def load_test_entity_name_map(entity_type: str):
    chem_desc = pd.read_csv('processed/description/chemical_description.csv')
    gene_desc = pd.read_csv('processed/description/gene_description.csv')
    dis_desc = pd.read_csv('processed/description/disease_description.csv')
    pheno_desc = pd.read_csv('processed/description/phenotype_description.csv')
    path_desc = pd.read_csv('processed/description/pathway_description.csv')
    go_desc = pd.read_csv('processed/description/go_description.csv')
    
    chem_name_map = chem_desc[['ChemicalID', 'ChemicalName']].set_index('ChemicalID').to_dict()['ChemicalName']
    gene_name_map = gene_desc[['GeneID', 'GeneName']].set_index('GeneID').to_dict()['GeneName']
    dis_name_map = dis_desc[['DiseaseID', 'DiseaseName']].set_index('DiseaseID').to_dict()['DiseaseName']
    pheno_name_map = pheno_desc[['PhenotypeID', 'PhenotypeName']].set_index('PhenotypeID').to_dict()['PhenotypeName']
    path_name_map = path_desc[['PathwayID', 'PathwayName']].set_index('PathwayID').to_dict()['PathwayName']
    go_name_map = go_desc[['GOTermID', 'GOTermName']].set_index('GOTermID').to_dict()['GOTermName']
    
    name_maps = (chem_name_map, gene_name_map, dis_name_map, pheno_name_map, path_name_map, go_name_map)
    
    # cd
    cd_chem_id_map = torch.load('processed/cd/chem_map')
    cd_dis_id_map = torch.load('processed/cd/dis_map')
    cd_test = torch.load('processed/cd/test_cd.pt')
    #
    cd_inv_chem_id_map = {v: k for k, v in cd_chem_id_map.items()}
    cd_inv_dis_id_map = {v: k for k, v in cd_dis_id_map.items()}
    #
    cd_test_chem_ids = extract_unique_ids(cd_test.edge_index_dict, 'chemical')
    cd_test_dis_ids = extract_unique_ids(cd_test.edge_index_dict, 'disease')
    #
    cd_test_chem_names = [cd_inv_chem_id_map[i] for i in cd_test_chem_ids]
    cd_test_dis_names = [cd_inv_dis_id_map[i] for i in cd_test_dis_ids]
    
    # cgd    
    cgd_chem_id_map = torch.load('processed/cgd/chem_map')
    cgd_gene_id_map = torch.load('processed/cgd/gene_map')
    cgd_dis_id_map = torch.load('processed/cgd/dis_map')
    cgd_test = torch.load('processed/cgd/test_cgd.pt')
    #
    cgd_inv_chem_id_map = {v: k for k, v in cgd_chem_id_map.items()}
    cgd_inv_gene_id_map = {v: k for k, v in cgd_gene_id_map.items()}
    cgd_inv_dis_id_map = {v: k for k, v in cgd_dis_id_map.items()}
    #
    cgd_test_chem_ids = extract_unique_ids(cgd_test.edge_index_dict, 'chemical')
    cgd_test_gene_ids = extract_unique_ids(cgd_test.edge_index_dict, 'gene')
    cgd_test_dis_ids = extract_unique_ids(cgd_test.edge_index_dict, 'disease')
    #
    cgd_test_chem_names = [cgd_inv_chem_id_map[i] for i in cgd_test_chem_ids]
    cgd_test_gene_names = [cgd_inv_gene_id_map[i] for i in cgd_test_gene_ids]
    cgd_test_dis_names = [cgd_inv_dis_id_map[i] for i in cgd_test_dis_ids]

    # cgpd
    cgpd_chem_id_map = torch.load('processed/cgpd/chem_map')
    cgpd_gene_id_map = torch.load('processed/cgpd/gene_map')
    cgpd_dis_id_map = torch.load('processed/cgpd/dis_map')
    cgpd_pheno_id_map = torch.load('processed/cgpd/pheno_map')
    cgpd_test = torch.load('processed/cgpd/test_cgpd.pt')
    #
    cgpd_inv_chem_id_map = {v: k for k, v in cgpd_chem_id_map.items()}
    cgpd_inv_gene_id_map = {v: k for k, v in cgpd_gene_id_map.items()}
    cgpd_inv_dis_id_map = {v: k for k, v in cgpd_dis_id_map.items()}
    cgpd_inv_pheno_id_map = {v: k for k, v in cgpd_pheno_id_map.items()}
    #
    cgpd_test_chem_ids = extract_unique_ids(cgpd_test.edge_index_dict, 'chemical')
    cgpd_test_gene_ids = extract_unique_ids(cgpd_test.edge_index_dict, 'gene')
    cgpd_test_pheno_ids = extract_unique_ids(cgpd_test.edge_index_dict, 'phenotype')
    cgpd_test_dis_ids = extract_unique_ids(cgpd_test.edge_index_dict, 'disease')
    #
    cgpd_test_chem_names = [cgpd_inv_chem_id_map[i] for i in cgpd_test_chem_ids]
    cgpd_test_gene_names = [cgpd_inv_gene_id_map[i] for i in cgpd_test_gene_ids]
    cgpd_test_pheno_names = [cgpd_inv_pheno_id_map[i] for i in cgpd_test_pheno_ids]
    cgpd_test_dis_names = [cgpd_inv_dis_id_map[i] for i in cgpd_test_dis_ids]
    
    # ctd
    ctd_chem_id_map = torch.load('processed/ctd/chem_map')
    ctd_gene_id_map = torch.load('processed/ctd/gene_map')
    ctd_dis_id_map = torch.load('processed/ctd/dis_map')
    ctd_pheno_id_map = torch.load('processed/ctd/pheno_map')
    ctd_path_id_map = torch.load('processed/ctd/path_map')
    ctd_go_id_map = torch.load('processed/ctd/go_map')
    ctd_test = torch.load('processed/ctd/test_ctd.pt')
    #
    ctd_inv_chem_id_map = {v: k for k, v in ctd_chem_id_map.items()}
    ctd_inv_gene_id_map = {v: k for k, v in ctd_gene_id_map.items()}
    ctd_inv_dis_id_map = {v: k for k, v in ctd_dis_id_map.items()}
    ctd_inv_pheno_id_map = {v: k for k, v in ctd_pheno_id_map.items()}
    ctd_inv_path_id_map = {v: k for k, v in ctd_path_id_map.items()}
    ctd_inv_go_id_map = {v: k for k, v in ctd_go_id_map.items()}
    #
    ctd_test_chem_ids = extract_unique_ids(ctd_test.edge_index_dict, 'chemical')
    ctd_test_gene_ids = extract_unique_ids(ctd_test.edge_index_dict, 'gene')
    ctd_test_dis_ids = extract_unique_ids(ctd_test.edge_index_dict, 'disease')
    ctd_test_pheno_ids = extract_unique_ids(ctd_test.edge_index_dict, 'phenotype')
    ctd_test_path_ids = extract_unique_ids(ctd_test.edge_index_dict, 'pathway')
    ctd_test_go_ids = extract_unique_ids(ctd_test.edge_index_dict, 'gene_ontology')
    #
    ctd_test_chem_names = [ctd_inv_chem_id_map[i] for i in ctd_test_chem_ids]
    ctd_test_gene_names = [ctd_inv_gene_id_map[i] for i in ctd_test_gene_ids]
    ctd_test_dis_names = [ctd_inv_dis_id_map[i] for i in ctd_test_dis_ids]
    ctd_test_pheno_names = [ctd_inv_pheno_id_map[i] for i in ctd_test_pheno_ids]
    ctd_test_path_names = [ctd_inv_path_id_map[i] for i in ctd_test_path_ids]
    ctd_test_go_names = [ctd_inv_go_id_map[i] for i in ctd_test_go_ids]
    
    ''' test에 속한 chemicals, genes, diseases, phenotyes, gos, pathways에 대해 ctd 기준 id mapping 만들기 '''
    test_chem_names = set(cd_test_chem_names + cgd_test_chem_names + cgpd_test_chem_names + ctd_test_chem_names)
    test_gene_names = set(cgd_test_gene_names + cgpd_test_gene_names + ctd_test_gene_names)
    test_dis_names = set(cd_test_dis_names + cgd_test_dis_names + cgpd_test_dis_names + ctd_test_dis_names)
    test_pheno_names = set(cgpd_test_pheno_names + ctd_test_pheno_names)
    test_path_names = set(ctd_test_path_names)
    test_go_names = set(ctd_test_go_names)

    chem_id_map = {k: ctd_chem_id_map[k] for k in test_chem_names}
    gene_id_map = {k: ctd_gene_id_map[k] for k in test_gene_names}
    dis_id_map = {k: ctd_dis_id_map[k] for k in test_dis_names}
    pheno_id_map = {k: ctd_pheno_id_map[k] for k in test_pheno_names}
    path_id_map = {k: ctd_path_id_map[k] for k in test_path_names}
    go_id_map = {k: ctd_go_id_map[k] for k in test_go_names}
    
    id_maps = (chem_id_map, gene_id_map, dis_id_map, pheno_id_map, path_id_map, go_id_map)
    
    inv_chem_map = {v: k for k, v in chem_id_map.items()}
    inv_gene_map = {v: k for k, v in gene_id_map.items()}
    inv_dis_map = {v: k for k, v in dis_id_map.items()}
    inv_pheno_map = {v: k for k, v in pheno_id_map.items()}
    inv_path_map = {v: k for k, v in path_id_map.items()}
    inv_go_map = {v: k for k, v in go_id_map.items()}
    
    inv_id_maps = (inv_chem_map, inv_gene_map, inv_dis_map, inv_pheno_map, inv_path_map, inv_go_map)
    
    if entity_type == 'chemical':
        return chem_desc, name_maps, id_maps, inv_id_maps
    elif entity_type == 'gene':
        return gene_desc, name_maps, id_maps, inv_id_maps
    elif entity_type == 'disease':
        return dis_desc, name_maps, id_maps, inv_id_maps
    elif entity_type == 'phenotype':
        return pheno_desc, name_maps, id_maps, inv_id_maps
    elif entity_type == 'pathway':
        return path_desc, name_maps, id_maps, inv_id_maps
    elif entity_type == 'go':
        return go_desc, name_maps, id_maps, inv_id_maps


def filter_description(row, head_entity, have_to_removed):
    if head_entity == 'chemical':
        head_col_name = f'{head_entity.title()}ID'
    elif head_entity == 'gene':
        head_col_name = f'{head_entity.title()}ID'
    elif head_entity == 'disease':
        head_col_name = f'{head_entity.title()}ID'
    elif head_entity == 'pathway':
        head_col_name = f'{head_entity.title()}ID'
    elif head_entity == 'phenotype':
        head_col_name = f'{head_entity.title()}ID'
    elif head_entity == 'go':
        head_col_name = 'GOTermID'
    
    head_id = row[head_col_name]
    text = row['MergedDescription']
    
    if pd.notna(text):  # NaN이 아닌 경우
        if head_entity == 'chemical':
            text += '.'
        else: pass
        words_to_remove = have_to_removed.get(head_id, [])
        if words_to_remove:  # 삭제할 단어가 존재하는 경우
            pattern = r'([^.!?]*\b(?:' + '|'.join(map(re.escape, words_to_remove)) + r')[^.!?]*[.!?])'
            text = re.sub(pattern, '', text, flags=re.IGNORECASE).strip()
            if text == '.': return np.nan
            elif text == '': return np.nan
            else: return text
        else:
            return text
    else: return text


def summarize_description(client, description, removed_dict):
    if pd.notna(description):
        try:
            prompt = (
                'Summarize the following text into no more than two concise sentences while preserving all key factual information. '
                'Do not adding or generating any new information not present in the original text. '
                'If the original text contains two sentences or fewer, return it unchanged. '
                'Also, remove any keywords listed here from the summary: {}.'
            ).format(', '.join(removed_dict))

            response = client.chat.completions.create(
                model='gpt-4o-mini',
                messages=[
                    {'role': 'system', 'content': prompt},
                    {'role': 'user', 'content': description}
                ],
                temperature=0,
                max_tokens=200
            )

            summarized_description = response.choices[0].message.content

        except:
            prompt = (
                'Summarize the following text into no more than two concise sentences while preserving all key factual information. '
                'Do not adding or generating any new information not present in the original text. '
                'If the original text contains two sentences or fewer, return it unchanged. '
            ).format(', '.join(removed_dict))

            response = client.chat.completions.create(
                model='gpt-4o-mini',
                messages=[
                    {'role': 'system', 'content': prompt},
                    {'role': 'user', 'content': description}
                ],
                temperature=0,
                max_tokens=200
            )

            summarized_description = response.choices[0].message.content
            
        return summarized_description
    else:
        return description


def preprocess_chemical_description(chem_desc, name_maps, id_maps, inv_id_maps, all_inv_id_maps, ctdkg):
    print('>>> Preprocessing Chemical Descriptions')
    
    # entity id -> entity name / entity id -> mapping id / mapping id -> entity id
    chem_name_map, gene_name_map, dis_name_map, pheno_name_map, path_name_map, go_name_map = name_maps
    chem_id_map, gene_id_map, dis_id_map, pheno_id_map, path_id_map, go_id_map = id_maps 
    inv_chem_map, inv_gene_map, inv_dis_map, inv_pheno_map, inv_path_map, inv_go_map = inv_id_maps
    all_inv_chem_map, all_inv_gene_map, all_inv_dis_map, all_inv_pheno_map, all_inv_path_map, all_inv_go_map = all_inv_id_maps
    
    # Extracting test data index
    test_idx = chem_desc.index[chem_desc.iloc[:, 0].isin( list(chem_id_map.keys()))]
    
    # Filtering the triplet according to the head
    chem_triplet = {(h, r, t): edges for (h, r, t), edges in ctdkg.edge_index_dict.items() if h == 'chemical'}
    
    # 각 head entity 마다 true triplet을 형성하는 tail entity 찾기
    have_to_removed = {k: [] for k in chem_id_map.keys()}
    
    tail_entity = list(set(t for (h, r, t) in chem_triplet.keys()))
    for tail in tqdm(tail_entity):
        if tail == 'chemical':
            tail_id_map = all_inv_chem_map; tail_name_map = chem_name_map
        elif tail == 'gene':
            tail_id_map = all_inv_gene_map; tail_name_map = gene_name_map
        elif tail == 'disease':
            tail_id_map = all_inv_dis_map; tail_name_map = dis_name_map
        elif tail == 'phenotype':
            tail_id_map = all_inv_pheno_map; tail_name_map = pheno_name_map
        elif tail == 'gene_ontology':
            tail_id_map = all_inv_go_map; tail_name_map = go_name_map
        elif tail == 'pathway':
            tail_id_map = all_inv_path_map; tail_name_map = path_name_map
        
        tmp = {(h, r, t): v for (h, r, t), v in chem_triplet.items() if t == tail}
        edges = torch.cat([edge for k, edge in tmp.items()], dim = 1).numpy()
        edges = np.unique(edges, axis = 1)
        
        head_to_id = np.vectorize(inv_chem_map.get)(edges[0])
        tail_to_id = np.vectorize(tail_id_map.get)(edges[1])
        
        for head_id, tail_id in zip(head_to_id, tail_to_id):
            if head_id in have_to_removed:
                tail_name = str(tail_name_map[tail_id])
                have_to_removed[head_id].append(tail_name if type(tail_name) == float else tail_name.lower())
    
    save_path = 'processed/description'
    if os.path.isdir(save_path):
        pass
    else:
        os.makedirs(save_path)
    
    save_path = os.path.join(save_path, 'chemical_description.csv')
    
    print('Summarzing the descriptions into no more than two sentences...')
    client = openai.OpenAI(api_key = open('openai_key', 'r').readline())
    
    # for i, description in enumerate(tqdm(chem_desc.FilteredDescription)):
    for i in tqdm(test_idx):
        description = chem_desc.FilteredDescription.iloc[i]
        remove_list = have_to_removed[chem_desc.ChemicalID.iloc[i]]
        
        # 요약 작업 수행
        chem_desc.at[i, 'SummarizedDescription'] = summarize_description(client, description, remove_list)

        # 일정 간격마다 중간 저장
        if i % 10 == 0 or i == len(chem_desc) - 1:
            chem_desc.to_csv(save_path, header = True, index=False)
            # print(f"Checkpoint saved at index {i} to {save_path}")
    
    print('>>> Preprocessing has successfully ended.')



def preprocess_gene_description(gene_desc, name_maps, id_maps, inv_id_maps, all_inv_id_maps, ctdkg):
    print('>>> Preprocessing Gene Descriptions')
    
    # entity id -> entity name / entity id -> mapping id / mapping id -> entity id
    chem_name_map, gene_name_map, dis_name_map, pheno_name_map, path_name_map, go_name_map = name_maps
    chem_id_map, gene_id_map, dis_id_map, pheno_id_map, path_id_map, go_id_map = id_maps 
    inv_chem_map, inv_gene_map, inv_dis_map, inv_pheno_map, inv_path_map, inv_go_map = inv_id_maps
    all_inv_chem_map, all_inv_gene_map, all_inv_dis_map, all_inv_pheno_map, all_inv_path_map, all_inv_go_map = all_inv_id_maps
    
    # Extracting test data index
    test_idx = gene_desc.index[gene_desc.iloc[:, 0].isin(gene_id_map.keys())]
    
    # Filtering the triplet according to the head
    gene_triplet = {(h, r, t): edges for (h, r, t), edges in ctdkg.edge_index_dict.items() if h == 'gene'}
    
    # 각 head entity 마다 true triplet을 형성하는 tail entity 찾기
    have_to_removed = {k: [] for k in gene_id_map.keys()}
    
    tail_entity = list(set(t for (h, r, t) in gene_triplet.keys()))
    for tail in tqdm(tail_entity):
        if tail == 'chemical':
            tail_id_map = all_inv_chem_map; tail_name_map = chem_name_map
        elif tail == 'gene':
            tail_id_map = all_inv_gene_map; tail_name_map = gene_name_map
        elif tail == 'disease':
            tail_id_map = all_inv_dis_map; tail_name_map = dis_name_map
        elif tail == 'phenotype':
            tail_id_map = all_inv_pheno_map; tail_name_map = pheno_name_map
        elif tail == 'gene_ontology':
            tail_id_map = all_inv_go_map; tail_name_map = go_name_map
        elif tail == 'pathway':
            tail_id_map = all_inv_path_map; tail_name_map = path_name_map
        
        tmp = {(h, r, t): v for (h, r, t), v in gene_triplet.items() if t == tail}
        edges = torch.cat([edge for k, edge in tmp.items()], dim = 1).numpy()
        edges = np.unique(edges, axis = 1)
        
        head_to_id = np.vectorize(all_inv_gene_map.get)(edges[0])
        tail_to_id = np.vectorize(tail_id_map.get)(edges[1])
        
        for head_id, tail_id in zip(head_to_id, tail_to_id):
            if head_id in have_to_removed:
                tail_name = str(tail_name_map[tail_id])
                have_to_removed[head_id].append(tail_name if type(tail_name) == float else tail_name.lower())
    
    save_path = 'processed/description'
    if os.path.isdir(save_path):
        pass
    else:
        os.makedirs(save_path)
    
    save_path = os.path.join(save_path, 'gene_description.csv')
    
    print('Summarzing the descriptions into no more than two sentences...')
    client = openai.OpenAI(api_key = open('openai_key', 'r').readline())
    
    for i in tqdm(test_idx):
        description = gene_desc.FilteredDescription.iloc[i]
        remove_list = have_to_removed[gene_desc.GeneID.iloc[i]]
        
        # 요약 작업 수행
        gene_desc.at[i, 'SummarizedDescription'] = summarize_description(client, description, remove_list)

        # 일정 간격마다 중간 저장
        if i % 10 > 7 or i == len(gene_desc) - 1:
            gene_desc.to_csv(save_path, header = True, index=False)
            # print(f"Checkpoint saved at index {i} to {save_path}")
    
    print('>>> Preprocessing has successfully ended.')



def preprocess_disease_description(dis_desc, name_maps, id_maps, inv_id_maps, all_inv_id_maps, ctdkg):
    print('>>> Preprocessing Disease Descriptions')
    
    # entity id -> entity name
    chem_name_map, gene_name_map, dis_name_map, pheno_name_map, path_name_map, go_name_map = name_maps
    chem_id_map, gene_id_map, dis_id_map, pheno_id_map, path_id_map, go_id_map = id_maps 
    inv_chem_map, inv_gene_map, inv_dis_map, inv_pheno_map, inv_path_map, inv_go_map = inv_id_maps
    all_inv_chem_map, all_inv_gene_map, all_inv_dis_map, all_inv_pheno_map, all_inv_path_map, all_inv_go_map = all_inv_id_maps
    
    # Extracting test data index
    test_idx = dis_desc.index[dis_desc.iloc[:, 0].isin(list(dis_id_map.keys()))]
    
    # Filtering the triplet according to the head
    disease_triplet = {(h, r, t): edges for (h, r, t), edges in ctdkg.edge_index_dict.items() if h == 'disease'}
    
    # 각 head entity 마다 true triplet을 형성하는 tail entity 찾기
    have_to_removed = {k: [] for k in dis_id_map.keys()}
    
    tail_entity = list(set(t for (h, r, t) in disease_triplet.keys()))
    for tail in tqdm(tail_entity):
        if tail == 'chemical':
            tail_id_map = all_inv_chem_map; tail_name_map = chem_name_map
        elif tail == 'gene':
            tail_id_map = all_inv_gene_map; tail_name_map = gene_name_map
        elif tail == 'disease':
            tail_id_map = all_inv_dis_map; tail_name_map = dis_name_map
        elif tail == 'phenotype':
            tail_id_map = all_inv_pheno_map; tail_name_map = pheno_name_map
        elif tail == 'gene_ontology':
            tail_id_map = all_inv_go_map; tail_name_map = go_name_map
        elif tail == 'pathway':
            tail_id_map = all_inv_path_map; tail_name_map = path_name_map
        
        tmp = {(h, r, t): v for (h, r, t), v in disease_triplet.items() if t == tail}
        edges = torch.cat([edge for k, edge in tmp.items()], dim = 1).numpy()
        edges = np.unique(edges, axis = 1)
        
        head_to_id = np.vectorize(inv_dis_map.get)(edges[0])
        tail_to_id = np.vectorize(tail_id_map.get)(edges[1])
        
        for head_id, tail_id in zip(head_to_id, tail_to_id):
            if head_id in have_to_removed:
                tail_name = str(tail_name_map[tail_id])
                have_to_removed[head_id].append(tail_name if type(tail_name) == float else tail_name.lower())
    
    save_path = 'processed/description'
    if os.path.isdir(save_path):
        pass
    else:
        os.makedirs(save_path)
    
    save_path = os.path.join(save_path, 'disease_description.csv')
    
    print('Summarzing the descriptions into no more than two sentences...')
    client = openai.OpenAI(api_key = open('openai_key', 'r').readline())
    
    for i in tqdm(test_idx):
        description = dis_desc.FilteredDescription.iloc[i]
        remove_list = have_to_removed[dis_desc.DiseaseID.iloc[i]]
        
        # 요약 작업 수행
        dis_desc.at[i, 'SummarizedDescription'] = summarize_description(client, description, remove_list)

        # 일정 간격마다 중간 저장
        if i % 10 == 0 or i == len(dis_desc) - 1:
            dis_desc.to_csv(save_path, header = True, index=False)
            # print(f"Checkpoint saved at index {i} to {save_path}")
    
    print('>>> Preprocessing has successfully ended.')



def preprocess_phenotype_description(pheno_desc, name_maps, id_maps, inv_id_maps, all_inv_id_maps, ctdkg):
    print('>>> Preprocessing Phenotype Descriptions')
    
    # entity id -> entity name
    chem_name_map, gene_name_map, dis_name_map, pheno_name_map, path_name_map, go_name_map = name_maps
    chem_id_map, gene_id_map, dis_id_map, pheno_id_map, path_id_map, go_id_map = id_maps 
    inv_chem_map, inv_gene_map, inv_dis_map, inv_pheno_map, inv_path_map, inv_go_map = inv_id_maps
    all_inv_chem_map, all_inv_gene_map, all_inv_dis_map, all_inv_pheno_map, all_inv_path_map, all_inv_go_map = all_inv_id_maps
    
    # Extracting test data index
    test_idx = pheno_desc.index[pheno_desc.iloc[:, 0].isin(list(pheno_id_map.keys()))]
    
    # Filtering the triplet according to the head
    pheno_triplet = {(h, r, t): edges for (h, r, t), edges in ctdkg.edge_index_dict.items() if h == 'phenotype'}
    
    # 각 head entity 마다 true triplet을 형성하는 tail entity 찾기
    have_to_removed = {k: [] for k in pheno_id_map.keys()}
    
    tail_entity = list(set(t for (h, r, t) in pheno_triplet.keys()))
    for tail in tqdm(tail_entity):
        if tail == 'chemical':
            tail_id_map = all_inv_chem_map; tail_name_map = chem_name_map
        elif tail == 'gene':
            tail_id_map = all_inv_gene_map; tail_name_map = gene_name_map
        elif tail == 'disease':
            tail_id_map = all_inv_dis_map; tail_name_map = dis_name_map
        elif tail == 'phenotype':
            tail_id_map = all_inv_pheno_map; tail_name_map = pheno_name_map
        elif tail == 'gene_ontology':
            tail_id_map = all_inv_go_map; tail_name_map = go_name_map
        elif tail == 'pathway':
            tail_id_map = all_inv_path_map; tail_name_map = path_name_map
        
        tmp = {(h, r, t): v for (h, r, t), v in pheno_triplet.items() if t == tail}
        edges = torch.cat([edge for k, edge in tmp.items()], dim = 1).numpy()
        edges = np.unique(edges, axis = 1)
        
        head_to_id = np.vectorize(inv_pheno_map.get)(edges[0])
        tail_to_id = np.vectorize(tail_id_map.get)(edges[1])
        
        for head_id, tail_id in zip(head_to_id, tail_to_id):
            if head_id in have_to_removed:
                tail_name = str(tail_name_map[tail_id])
                have_to_removed[head_id].append(tail_name if type(tail_name) == float else tail_name.lower())
    
    save_path = 'processed/description'
    if os.path.isdir(save_path):
        pass
    else:
        os.makedirs(save_path)
    
    save_path = os.path.join(save_path, 'phenotype_description.csv')
    
    print('Summarzing the descriptions into no more than two sentences...')
    client = openai.OpenAI(api_key = open('openai_key', 'r').readline())
    
    for i in tqdm(test_idx):
        description = pheno_desc.FilteredDescription.iloc[i]
        remove_list = have_to_removed[pheno_desc.PhenotypeID.iloc[i]]
        
        # 요약 작업 수행
        pheno_desc.at[i, 'SummarizedDescription'] = summarize_description(client, description, remove_list)

        # 일정 간격마다 중간 저장
        if i % 10 > 7 or i == len(pheno_desc) - 1:
            pheno_desc.to_csv(save_path, header = True, index=False)
            # print(f"Checkpoint saved at index {i} to {save_path}")
    
    print('>>> Preprocessing has successfully ended.')
    


def preprocess_pathway_description(path_desc, name_maps, id_maps, inv_id_maps, all_inv_id_maps, ctdkg):
    print('>>> Preprocessing Pathway Descriptions')
    
    # entity id -> entity name
    chem_name_map, gene_name_map, dis_name_map, pheno_name_map, path_name_map, go_name_map = name_maps
    chem_id_map, gene_id_map, dis_id_map, pheno_id_map, path_id_map, go_id_map = id_maps 
    inv_chem_map, inv_gene_map, inv_dis_map, inv_pheno_map, inv_path_map, inv_go_map = inv_id_maps
    all_inv_chem_map, all_inv_gene_map, all_inv_dis_map, all_inv_pheno_map, all_inv_path_map, all_inv_go_map = all_inv_id_maps
    
    # Extracting test data index
    test_idx = path_desc.index[path_desc.iloc[:, 0].isin(list(path_id_map.keys()))]
    
    # Filtering the triplet according to the head
    path_triplet = {(h, r, t): edges for (h, r, t), edges in ctdkg.edge_index_dict.items() if h == 'pathway'}
    
    # 각 head entity 마다 true triplet을 형성하는 tail entity 찾기
    have_to_removed = {k: [] for k in path_id_map.keys()}
    
    tail_entity = list(set(t for (h, r, t) in path_triplet.keys()))
    for tail in tqdm(tail_entity):
        if tail == 'chemical':
            tail_id_map = all_inv_chem_map; tail_name_map = chem_name_map
        elif tail == 'gene':
            tail_id_map = all_inv_gene_map; tail_name_map = gene_name_map
        elif tail == 'disease':
            tail_id_map = all_inv_dis_map; tail_name_map = dis_name_map
        elif tail == 'phenotype':
            tail_id_map = all_inv_pheno_map; tail_name_map = pheno_name_map
        elif tail == 'gene_ontology':
            tail_id_map = all_inv_go_map; tail_name_map = go_name_map
        elif tail == 'pathway':
            tail_id_map = all_inv_path_map; tail_name_map = path_name_map
        
        tmp = {(h, r, t): v for (h, r, t), v in path_triplet.items() if t == tail}
        edges = torch.cat([edge for k, edge in tmp.items()], dim = 1).numpy()
        edges = np.unique(edges, axis = 1)
        
        head_to_id = np.vectorize(inv_path_map.get)(edges[0])
        tail_to_id = np.vectorize(tail_id_map.get)(edges[1])
        
        for head_id, tail_id in zip(head_to_id, tail_to_id):
            if head_id in have_to_removed:
                tail_name = str(tail_name_map[tail_id])
                have_to_removed[head_id].append(tail_name if type(tail_name) == float else tail_name.lower())
    
    save_path = 'processed/description'
    if os.path.isdir(save_path):
        pass
    else:
        os.makedirs(save_path)
    
    save_path = os.path.join(save_path, 'pathway_description.csv')
    
    print('Summarzing the descriptions into no more than two sentences...')
    client = openai.OpenAI(api_key = open('openai_key', 'r').readline())
        
    for i in tqdm(test_idx):
        description = path_desc.FilteredDescription.iloc[i]
        remove_list = have_to_removed[path_desc.PathwayID.iloc[i]]
        
        # 요약 작업 수행
        path_desc.at[i, 'SummarizedDescription'] = summarize_description(client, description, remove_list)

        # 일정 간격마다 중간 저장
        if i % 10 > 7 or i == len(path_desc) - 1:
            path_desc.to_csv(save_path, header = True, index=False)
            # print(f"Checkpoint saved at index {i} to {save_path}")
    
    print('>>> Preprocessing has successfully ended.')



def preprocess_go_description(go_desc, name_maps, id_maps, inv_id_maps, all_inv_id_maps, ctdkg):
    print('>>> Preprocessing Gene Ontology Descriptions')
    
    # entity id -> entity name
    chem_name_map, gene_name_map, dis_name_map, pheno_name_map, path_name_map, go_name_map = name_maps
    chem_id_map, gene_id_map, dis_id_map, pheno_id_map, path_id_map, go_id_map = id_maps 
    inv_chem_map, inv_gene_map, inv_dis_map, inv_pheno_map, inv_path_map, inv_go_map = inv_id_maps
    all_inv_chem_map, all_inv_gene_map, all_inv_dis_map, all_inv_pheno_map, all_inv_path_map, all_inv_go_map = all_inv_id_maps
    
    # Extracting test data index
    test_idx = go_desc.index[go_desc.iloc[:, 0].isin(list(go_id_map.keys()))]
    
    # Filtering the triplet according to the head
    go_triplet = {(h, r, t): edges for (h, r, t), edges in ctdkg.edge_index_dict.items() if h == 'gene_ontology'}
    
    # 각 head entity 마다 true triplet을 형성하는 tail entity 찾기
    have_to_removed = {k: [] for k in go_id_map.keys()}
    
    tail_entity = list(set(t for (h, r, t) in go_triplet.keys()))
    for tail in tqdm(tail_entity):
        if tail == 'chemical':
            tail_id_map = all_inv_chem_map; tail_name_map = chem_name_map
        elif tail == 'gene':
            tail_id_map = all_inv_gene_map; tail_name_map = gene_name_map
        elif tail == 'disease':
            tail_id_map = all_inv_dis_map; tail_name_map = dis_name_map
        elif tail == 'phenotype':
            tail_id_map = all_inv_pheno_map; tail_name_map = pheno_name_map
        elif tail == 'gene_ontology':
            tail_id_map = all_inv_go_map; tail_name_map = go_name_map
        elif tail == 'pathway':
            tail_id_map = all_inv_path_map; tail_name_map = path_name_map
        
        tmp = {(h, r, t): v for (h, r, t), v in go_triplet.items() if t == tail}
        edges = torch.cat([edge for k, edge in tmp.items()], dim = 1).numpy()
        edges = np.unique(edges, axis = 1)
        
        head_to_id = np.vectorize(inv_go_map.get)(edges[0])
        tail_to_id = np.vectorize(tail_id_map.get)(edges[1])
        
        for head_id, tail_id in zip(head_to_id, tail_to_id):
            if head_id in have_to_removed:
                tail_name = str(tail_name_map[tail_id])
                have_to_removed[head_id].append(tail_name if type(tail_name) == float else tail_name.lower())
    
    save_path = 'processed/description'
    if os.path.isdir(save_path):
        pass
    else:
        os.makedirs(save_path)
    
    save_path = os.path.join(save_path, 'go_description.csv')
    
    print('Summarzing the descriptions into no more than two sentences...')
    client = openai.OpenAI(api_key = open('openai_key', 'r').readline())
    for i in tqdm(test_idx):
        description = go_desc.FilteredDescription.iloc[i]
        remove_list = have_to_removed[go_desc.GOTermID.iloc[i]]
        
        # 요약 작업 수행
        go_desc.at[i, 'SummarizedDescription'] = summarize_description(client, description, remove_list)

        # 일정 간격마다 중간 저장
        if i % 10 > 7 or i == len(go_desc) - 1:
            go_desc.to_csv(save_path, header = True, index=False)
            # print(f"Checkpoint saved at index {i} to {save_path}")
            
    print('>>> Preprocessing has successfully ended.')



def preprocess_description(entity_type: str):
    desc_df, name_maps, id_maps, inv_id_maps = load_test_entity_name_map(entity_type)
    _, all_inv_id_maps = load_entity_name_map(entity_type)
    ctdkg = torch.load('processed/ctd/ctd.pt')
    
    tqdm.pandas()
    if entity_type == 'chemical':
        preprocess_chemical_description(desc_df, name_maps, id_maps, inv_id_maps, all_inv_id_maps, ctdkg)
    elif entity_type == 'gene':
        preprocess_gene_description(desc_df, name_maps, id_maps, inv_id_maps, all_inv_id_maps, ctdkg)
    elif entity_type == 'disease':
        preprocess_disease_description(desc_df, name_maps, id_maps, inv_id_maps, all_inv_id_maps, ctdkg)
    elif entity_type == 'phenotype':
        preprocess_phenotype_description(desc_df, name_maps, id_maps, inv_id_maps, all_inv_id_maps, ctdkg)
    elif entity_type == 'pathway':
        preprocess_pathway_description(desc_df, name_maps, id_maps, inv_id_maps, all_inv_id_maps, ctdkg)
    elif entity_type == 'go':
        preprocess_go_description(desc_df, name_maps, id_maps, inv_id_maps, all_inv_id_maps, ctdkg)



if __name__ == '__main__':
    # preprocess_description('chemical')
    # preprocess_description('gene')
    # preprocess_description('disease')
    # preprocess_description('phenotype')
    preprocess_description('pathway')
    preprocess_description('go')
