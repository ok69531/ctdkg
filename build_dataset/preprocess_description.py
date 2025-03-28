import os
import re
import torch
import openai
import warnings

import numpy as np
import pandas as pd

from tqdm.autonotebook import tqdm

warnings.filterwarnings('ignore')


def load_entity_name_map(entity_type: str):
    chem_desc = pd.read_csv('raw/description/chem_description.csv')
    gene_desc = pd.read_csv('raw/description/gene_description.csv')
    dis_desc = pd.read_csv('raw/description/dis_description.csv')
    pheno_desc = pd.read_csv('raw/description/pheno_description.csv')
    path_desc = pd.read_csv('raw/description/pathway_description.csv')
    go_desc = pd.read_csv('raw/description/go_description.csv')
    
    chem_name_map = chem_desc[['ChemicalID', 'ChemicalName']].set_index('ChemicalID').to_dict()['ChemicalName']
    gene_name_map = gene_desc[['GeneID', 'GeneName']].set_index('GeneID').to_dict()['GeneName']
    dis_name_map = dis_desc[['DiseaseID', 'DiseaseName']].set_index('DiseaseID').to_dict()['DiseaseName']
    pheno_name_map = pheno_desc[['PhenotypeID', 'PhenotypeName']].set_index('PhenotypeID').to_dict()['PhenotypeName']
    path_name_map = path_desc[['PathwayID', 'PathwayName']].set_index('PathwayID').to_dict()['PathwayName']
    go_name_map = go_desc[['GOTermID', 'GOTermName']].set_index('GOTermID').to_dict()['GOTermName']
    
    name_maps = (chem_name_map, gene_name_map, dis_name_map, pheno_name_map, path_name_map, go_name_map)
    
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



def summarize_description(client, description):
    if pd.notna(description):
        prompt = 'Summarize the following text into no more than two concise sentences while preserving all key factual information. Avoid adding or generating any new information not present in the original text. If the original text contains two sentences or fewer, return it unchanged.'

        response = client.chat.completions.create(
            model = 'gpt-4o',
            messages = [
                    {'role': 'system', 'content': prompt},
                    {'role': 'user', 'content': description}
                ],
            temperature = 0,
            max_tokens = 100
        )

        summarized_description = response.choices[0].message.content
        
        return summarized_description
    else: return description



def preprocess_chemical_description(raw_chem_desc, name_maps, id_maps, inv_id_maps, ctdkg):
    print('>>> Preprocessing Chemical Descriptions')
    
    # Deleting the meaningless descriptions in MeSH DB
    dup_desc = raw_chem_desc['MeSHDescription'][raw_chem_desc['MeSHDescription'].duplicated(keep = False)]
    dup_desc = dup_desc.value_counts()[dup_desc.value_counts() > 10].index
    dup_idx = raw_chem_desc['MeSHDescription'].isin(dup_desc)
    raw_chem_desc['MeSHDescription'][dup_idx] = np.nan
    
    # Merging the multiple descriptions
    raw_chem_desc['MergedDescription'] = raw_chem_desc.iloc[:, 2:].apply(
        lambda x: np.nan if x.isna().all() else '. '.join(x.dropna()), axis = 1
    )
    
    # entity id -> entity name / mapping id -> entity id / entity id -> mapping id
    chem_name_map, gene_name_map, dis_name_map, pheno_name_map, path_name_map, go_name_map = name_maps
    chem_id_map, gene_id_map, dis_id_map, pheno_id_map, path_id_map, go_id_map = id_maps 
    inv_chem_map, inv_gene_map, inv_dis_map, inv_pheno_map, inv_path_map, inv_go_map = inv_id_maps
    
    # Filtering the triplet according to the head
    chem_triplet = {(h, r, t): edges for (h, r, t), edges in ctdkg.edge_index_dict.items() if h == 'chemical'}
    
    # 각 head entity 마다 true triplet을 형성하는 tail entity 찾기
    have_to_removed = {k: [] for k in chem_id_map.keys()}
    
    tail_entity = list(set(t for (h, r, t) in chem_triplet.keys()))
    for tail in tqdm(tail_entity):
        if tail == 'chemical':
            tail_id_map = inv_chem_map; tail_name_map = chem_name_map
        elif tail == 'gene':
            tail_id_map = inv_gene_map; tail_name_map = gene_name_map
        elif tail == 'disease':
            tail_id_map = inv_dis_map; tail_name_map = dis_name_map
        elif tail == 'phenotype':
            tail_id_map = inv_pheno_map; tail_name_map = pheno_name_map
        elif tail == 'gene_ontology':
            tail_id_map = inv_go_map; tail_name_map = go_name_map
        elif tail == 'pathway':
            tail_id_map = inv_path_map; tail_name_map = path_name_map
        
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
    
    # merged description에 true triplet 정보가 포함되는 경우 삭제 - rule based matching
    print('Removing the true relations in descriptions...')
    raw_chem_desc['FilteredDescription'] = raw_chem_desc.progress_apply(
        lambda row: filter_description(row, 'chemical', have_to_removed), axis=1
    )
    raw_chem_desc.to_csv(save_path, header = True, index=False)
    
    print('Summarzing the descriptions into no more than two sentences...')
    client = openai.OpenAI(api_key = open('openai_key', 'r').readline())
    # raw_chem_desc['SummarizedDescription'] = raw_chem_desc.FilteredDescription.progress_apply(
    #     lambda x: summarize_description(client, x)
    # )
    
    for i, description in enumerate(tqdm(raw_chem_desc.FilteredDescription)):
        # 요약 작업 수행
        raw_chem_desc.at[i, 'SummarizedDescription'] = summarize_description(client, description)

        # 일정 간격마다 중간 저장
        if i % 10 == 0 or i == len(raw_chem_desc) - 1:
            raw_chem_desc.to_csv(save_path, header = True, index=False)
            # print(f"Checkpoint saved at index {i} to {save_path}")
    
    print('>>> Preprocessing has successfully ended.')



def preprocess_gene_description(raw_gene_desc, name_maps, id_maps, inv_id_maps, ctdkg):
    print('>>> Preprocessing Gene Descriptions')
    # Merging the multiple descriptions
    raw_gene_desc['MergedDescription'] = raw_gene_desc.iloc[:, 2:].apply(
        lambda x: np.nan if x.isna().all() else '. '.join(x.dropna()), axis = 1
    )
    
    # entity id -> entity name
    chem_name_map, gene_name_map, dis_name_map, pheno_name_map, path_name_map, go_name_map = name_maps
    chem_id_map, gene_id_map, dis_id_map, pheno_id_map, path_id_map, go_id_map = id_maps 
    inv_chem_map, inv_gene_map, inv_dis_map, inv_pheno_map, inv_path_map, inv_go_map = inv_id_maps
    
    # Filtering the triplet according to the head
    gene_triplet = {(h, r, t): edges for (h, r, t), edges in ctdkg.edge_index_dict.items() if h == 'gene'}
    
    # 각 head entity 마다 true triplet을 형성하는 tail entity 찾기
    have_to_removed = {k: [] for k in gene_id_map.keys()}
    
    tail_entity = list(set(t for (h, r, t) in gene_triplet.keys()))
    for tail in tqdm(tail_entity):
        if tail == 'chemical':
            tail_id_map = inv_chem_map; tail_name_map = chem_name_map
        elif tail == 'gene':
            tail_id_map = inv_gene_map; tail_name_map = gene_name_map
        elif tail == 'disease':
            tail_id_map = inv_dis_map; tail_name_map = dis_name_map
        elif tail == 'phenotype':
            tail_id_map = inv_pheno_map; tail_name_map = pheno_name_map
        elif tail == 'gene_ontology':
            tail_id_map = inv_go_map; tail_name_map = go_name_map
        elif tail == 'pathway':
            tail_id_map = inv_path_map; tail_name_map = path_name_map
        
        tmp = {(h, r, t): v for (h, r, t), v in gene_triplet.items() if t == tail}
        edges = torch.cat([edge for k, edge in tmp.items()], dim = 1).numpy()
        edges = np.unique(edges, axis = 1)
        
        head_to_id = np.vectorize(inv_gene_map.get)(edges[0])
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
    
    # merged description에 true triplet 정보가 포함되는 경우 삭제 - rule based matching
    print('Removing the true relations in descriptions...')
    raw_gene_desc['FilteredDescription'] = raw_gene_desc.progress_apply(
        lambda row: filter_description(row, 'gene', have_to_removed), axis=1
    )
    raw_gene_desc.to_csv(save_path, header = True, index=False)
    
    print('Summarzing the descriptions into no more than two sentences...')
    client = openai.OpenAI(api_key = open('openai_key', 'r').readline())
    
    for i, description in enumerate(tqdm(raw_gene_desc.FilteredDescription)):
        # 요약 작업 수행
        raw_gene_desc.at[i, 'SummarizedDescription'] = summarize_description(client, description)

        # 일정 간격마다 중간 저장
        if i % 10 == 0 or i == len(raw_gene_desc) - 1:
            raw_gene_desc.to_csv(save_path, header = True, index=False)
            # print(f"Checkpoint saved at index {i} to {save_path}")
    
    print('>>> Preprocessing has successfully ended.')



def preprocess_disease_description(raw_dis_desc, name_maps, id_maps, inv_id_maps, ctdkg):
    print('>>> Preprocessing Disease Descriptions')
    # Merging the multiple descriptions
    raw_dis_desc['MergedDescription'] = raw_dis_desc.iloc[:, 2:].apply(
        lambda x: np.nan if x.isna().all() else '. '.join(x.dropna()), axis = 1
    )
    
    # entity id -> entity name
    chem_name_map, gene_name_map, dis_name_map, pheno_name_map, path_name_map, go_name_map = name_maps
    chem_id_map, gene_id_map, dis_id_map, pheno_id_map, path_id_map, go_id_map = id_maps 
    inv_chem_map, inv_gene_map, inv_dis_map, inv_pheno_map, inv_path_map, inv_go_map = inv_id_maps
    
    # Filtering the triplet according to the head
    disease_triplet = {(h, r, t): edges for (h, r, t), edges in ctdkg.edge_index_dict.items() if h == 'disease'}
    
    # 각 head entity 마다 true triplet을 형성하는 tail entity 찾기
    have_to_removed = {k: [] for k in dis_id_map.keys()}
    
    tail_entity = list(set(t for (h, r, t) in disease_triplet.keys()))
    for tail in tqdm(tail_entity):
        if tail == 'chemical':
            tail_id_map = inv_chem_map; tail_name_map = chem_name_map
        elif tail == 'gene':
            tail_id_map = inv_gene_map; tail_name_map = gene_name_map
        elif tail == 'disease':
            tail_id_map = inv_dis_map; tail_name_map = dis_name_map
        elif tail == 'phenotype':
            tail_id_map = inv_pheno_map; tail_name_map = pheno_name_map
        elif tail == 'gene_ontology':
            tail_id_map = inv_go_map; tail_name_map = go_name_map
        elif tail == 'pathway':
            tail_id_map = inv_path_map; tail_name_map = path_name_map
        
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
    
    # merged description에 true triplet 정보가 포함되는 경우 삭제 - rule based matching
    print('Removing the true relations in descriptions...')
    raw_dis_desc['FilteredDescription'] = raw_dis_desc.progress_apply(
        lambda row: filter_description(row, 'disease', have_to_removed), axis=1
    )
    raw_dis_desc.to_csv(save_path, header = True, index=False)
    
    print('Summarzing the descriptions into no more than two sentences...')
    client = openai.OpenAI(api_key = open('openai_key', 'r').readline())
    
    for i, description in enumerate(tqdm(raw_dis_desc.FilteredDescription)):
        # 요약 작업 수행
        raw_dis_desc.at[i, 'SummarizedDescription'] = summarize_description(client, description)

        # 일정 간격마다 중간 저장
        if i % 10 == 0 or i == len(raw_dis_desc) - 1:
            raw_dis_desc.to_csv(save_path, header = True, index=False)
            # print(f"Checkpoint saved at index {i} to {save_path}")
    
    print('>>> Preprocessing has successfully ended.')



def preprocess_phenotype_description(raw_pheno_desc, name_maps, id_maps, inv_id_maps, ctdkg):
    print('>>> Preprocessing Phenotype Descriptions')
    # Merging the multiple descriptions
    raw_pheno_desc['MergedDescription'] = raw_pheno_desc.iloc[:, 2:].apply(
        lambda x: np.nan if x.isna().all() else '. '.join(x.dropna()), axis = 1
    )
    
    # entity id -> entity name
    chem_name_map, gene_name_map, dis_name_map, pheno_name_map, path_name_map, go_name_map = name_maps
    chem_id_map, gene_id_map, dis_id_map, pheno_id_map, path_id_map, go_id_map = id_maps 
    inv_chem_map, inv_gene_map, inv_dis_map, inv_pheno_map, inv_path_map, inv_go_map = inv_id_maps
    
    # Filtering the triplet according to the head
    pheno_triplet = {(h, r, t): edges for (h, r, t), edges in ctdkg.edge_index_dict.items() if h == 'phenotype'}
    
    # 각 head entity 마다 true triplet을 형성하는 tail entity 찾기
    have_to_removed = {k: [] for k in pheno_id_map.keys()}
    
    tail_entity = list(set(t for (h, r, t) in pheno_triplet.keys()))
    for tail in tqdm(tail_entity):
        if tail == 'chemical':
            tail_id_map = inv_chem_map; tail_name_map = chem_name_map
        elif tail == 'gene':
            tail_id_map = inv_gene_map; tail_name_map = gene_name_map
        elif tail == 'disease':
            tail_id_map = inv_dis_map; tail_name_map = dis_name_map
        elif tail == 'phenotype':
            tail_id_map = inv_pheno_map; tail_name_map = pheno_name_map
        elif tail == 'gene_ontology':
            tail_id_map = inv_go_map; tail_name_map = go_name_map
        elif tail == 'pathway':
            tail_id_map = inv_path_map; tail_name_map = path_name_map
        
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
    
    # merged description에 true triplet 정보가 포함되는 경우 삭제 - rule based matching
    print('Removing the true relations in descriptions...')
    raw_pheno_desc['FilteredDescription'] = raw_pheno_desc.progress_apply(
        lambda row: filter_description(row, 'phenotype', have_to_removed), axis=1
    )
    raw_pheno_desc.to_csv(save_path, header = True, index=False)
    
    print('Summarzing the descriptions into no more than two sentences...')
    client = openai.OpenAI(api_key = open('openai_key', 'r').readline())
    
    for i, description in enumerate(tqdm(raw_pheno_desc.FilteredDescription)):
        # 요약 작업 수행
        raw_pheno_desc.at[i, 'SummarizedDescription'] = summarize_description(client, description)

        # 일정 간격마다 중간 저장
        if i % 10 == 0 or i == len(raw_pheno_desc) - 1:
            raw_pheno_desc.to_csv(save_path, header = True, index=False)
            # print(f"Checkpoint saved at index {i} to {save_path}")
    
    print('>>> Preprocessing has successfully ended.')
    


def preprocess_pathway_description(raw_path_desc, name_maps, id_maps, inv_id_maps, ctdkg):
    print('>>> Preprocessing Pathway Descriptions')
    # Merging the multiple descriptions
    raw_path_desc['MergedDescription'] = raw_path_desc.iloc[:, 2:].apply(
        lambda x: np.nan if x.isna().all() else '. '.join(x.dropna()), axis = 1
    )
    
    # entity id -> entity name
    chem_name_map, gene_name_map, dis_name_map, pheno_name_map, path_name_map, go_name_map = name_maps
    chem_id_map, gene_id_map, dis_id_map, pheno_id_map, path_id_map, go_id_map = id_maps 
    inv_chem_map, inv_gene_map, inv_dis_map, inv_pheno_map, inv_path_map, inv_go_map = inv_id_maps
    
    # Filtering the triplet according to the head
    path_triplet = {(h, r, t): edges for (h, r, t), edges in ctdkg.edge_index_dict.items() if h == 'pathway'}
    
    # 각 head entity 마다 true triplet을 형성하는 tail entity 찾기
    have_to_removed = {k: [] for k in path_id_map.keys()}
    
    tail_entity = list(set(t for (h, r, t) in path_triplet.keys()))
    for tail in tqdm(tail_entity):
        if tail == 'chemical':
            tail_id_map = inv_chem_map; tail_name_map = chem_name_map
        elif tail == 'gene':
            tail_id_map = inv_gene_map; tail_name_map = gene_name_map
        elif tail == 'disease':
            tail_id_map = inv_dis_map; tail_name_map = dis_name_map
        elif tail == 'phenotype':
            tail_id_map = inv_pheno_map; tail_name_map = pheno_name_map
        elif tail == 'gene_ontology':
            tail_id_map = inv_go_map; tail_name_map = go_name_map
        elif tail == 'pathway':
            tail_id_map = inv_path_map; tail_name_map = path_name_map
        
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
    
    # merged description에 true triplet 정보가 포함되는 경우 삭제 - rule based matching
    print('Removing the true relations in descriptions...')
    raw_path_desc['FilteredDescription'] = raw_path_desc.progress_apply(
        lambda row: filter_description(row, 'pathway', have_to_removed), axis=1
    )
    raw_path_desc.to_csv(save_path, header = True, index = False)
    
    print('Summarzing the descriptions into no more than two sentences...')
    client = openai.OpenAI(api_key = open('openai_key', 'r').readline())
        
    for i, description in enumerate(tqdm(raw_path_desc.FilteredDescription)):
        # 요약 작업 수행
        raw_path_desc.at[i, 'SummarizedDescription'] = summarize_description(client, description)

        # 일정 간격마다 중간 저장
        if i % 10 == 0 or i == len(raw_path_desc) - 1:
            raw_path_desc.to_csv(save_path, header = True, index=False)
            # print(f"Checkpoint saved at index {i} to {save_path}")
    
    print('>>> Preprocessing has successfully ended.')



def preprocess_go_description(raw_go_desc, name_maps, id_maps, inv_id_maps, ctdkg):
    print('>>> Preprocessing Gene Ontology Descriptions')
    # Merging the multiple descriptions
    raw_go_desc['MergedDescription'] = raw_go_desc.iloc[:, 2:].apply(
        lambda x: np.nan if x.isna().all() else '. '.join(x.dropna()), axis = 1
    )
    
    # entity id -> entity name
    chem_name_map, gene_name_map, dis_name_map, pheno_name_map, path_name_map, go_name_map = name_maps
    chem_id_map, gene_id_map, dis_id_map, pheno_id_map, path_id_map, go_id_map = id_maps 
    inv_chem_map, inv_gene_map, inv_dis_map, inv_pheno_map, inv_path_map, inv_go_map = inv_id_maps
    
    # Filtering the triplet according to the head
    go_triplet = {(h, r, t): edges for (h, r, t), edges in ctdkg.edge_index_dict.items() if h == 'gene_ontology'}
    
    # 각 head entity 마다 true triplet을 형성하는 tail entity 찾기
    have_to_removed = {k: [] for k in go_id_map.keys()}
    
    tail_entity = list(set(t for (h, r, t) in go_triplet.keys()))
    for tail in tqdm(tail_entity):
        if tail == 'chemical':
            tail_id_map = inv_chem_map; tail_name_map = chem_name_map
        elif tail == 'gene':
            tail_id_map = inv_gene_map; tail_name_map = gene_name_map
        elif tail == 'disease':
            tail_id_map = inv_dis_map; tail_name_map = dis_name_map
        elif tail == 'phenotype':
            tail_id_map = inv_pheno_map; tail_name_map = pheno_name_map
        elif tail == 'gene_ontology':
            tail_id_map = inv_go_map; tail_name_map = go_name_map
        elif tail == 'pathway':
            tail_id_map = inv_path_map; tail_name_map = path_name_map
        
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
    
    # merged description에 true triplet 정보가 포함되는 경우 삭제 - rule based matching
    print('Removing the true relations in descriptions...')
    raw_go_desc['FilteredDescription'] = raw_go_desc.progress_apply(
        lambda row: filter_description(row, 'go', have_to_removed), axis=1
    )
    raw_go_desc.to_csv(save_path, header = True, index = False)
    
    print('Summarzing the descriptions into no more than two sentences...')
    client = openai.OpenAI(api_key = open('openai_key', 'r').readline())
    for i, description in enumerate(tqdm(raw_go_desc.FilteredDescription)):
        # 요약 작업 수행
        raw_go_desc.at[i, 'SummarizedDescription'] = summarize_description(client, description)

        # 일정 간격마다 중간 저장
        if i % 10 == 0 or i == len(raw_go_desc) - 1:
            raw_go_desc.to_csv(save_path, header = True, index=False)
            # print(f"Checkpoint saved at index {i} to {save_path}")
            
    print('>>> Preprocessing has successfully ended.')



def preprocess_description(entity_type: str):
    raw_df, name_maps, id_maps, inv_id_maps = load_entity_name_map(entity_type)
    ctdkg = torch.load('processed/ctd/ctd.pt')
    
    tqdm.pandas()
    if entity_type == 'chemical':
        preprocess_chemical_description(raw_df, name_maps, id_maps, inv_id_maps, ctdkg)
    elif entity_type == 'gene':
        preprocess_gene_description(raw_df, name_maps, id_maps, inv_id_maps, ctdkg)
    elif entity_type == 'disease':
        preprocess_disease_description(raw_df, name_maps, id_maps, inv_id_maps, ctdkg)
    elif entity_type == 'phenotype':
        preprocess_phenotype_description(raw_df, name_maps, id_maps, inv_id_maps, ctdkg)
    elif entity_type == 'pathway':
        preprocess_pathway_description(raw_df, name_maps, id_maps, inv_id_maps, ctdkg)
    elif entity_type == 'go':
        preprocess_go_description(raw_df, name_maps, id_maps, inv_id_maps, ctdkg)



if __name__ == '__main__':
    preprocess_description('chemical')
    preprocess_description('gene')
    preprocess_description('disease')
    preprocess_description('phenotype')
    preprocess_description('pathway')
    preprocess_description('go')
