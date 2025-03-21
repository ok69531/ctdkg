import re
import pandas as pd
from tqdm import tqdm

import torch
# BioT5+
from transformers import T5Tokenizer, T5ForConditionalGeneration
# ChemDFM
from transformers import LlamaTokenizer, LlamaForCausalLM


def load_model(model_name):
    if model_name == 'biot5+':
        tokenizer = T5Tokenizer.from_pretrained('QizhiPei/biot5-plus-large', model_max_length = 512)
        model = T5ForConditionalGeneration.from_pretrained('QizhiPei/biot5-plus-large')

        generation_config = model.generation_config
        generation_config.max_length = 512
        generation_config.num_beams = 1
        
    elif model_name == 'chemdfm':
        model_name_or_id = "OpenDFM/ChemDFM-13B-v1.0"
        tokenizer = LlamaTokenizer.from_pretrained(model_name_or_id)
        model = LlamaForCausalLM.from_pretrained(model_name_or_id, torch_dtype=torch.float16, device_map="auto")
        
    return tokenizer, model
# tokenizer, model = load_model('chemdfm')
# input_ids = tokenizer(input_text, return_tensors = 'pt')
# a = model(**input_ids, output_hidden_state = True)


def chemical_embedding(model_name):
    tokenizer, model = load_model(model_name)
    model.eval()
    
    chem_desc = pd.read_csv('processed/description/chemical_description.csv')
    chem_desc['input_text'] = chem_desc['ChemicalName'].fillna('') + '. ' + chem_desc['SummarizedDescription'].fillna('')
    chem_desc['input_text'] = chem_desc['input_text'].apply(lambda x: x.strip())

    chem_emb = {x: None for x in chem_desc.ChemicalID}
    for i in tqdm(range(len(chem_desc))):
        chem_id = chem_desc.ChemicalID[i]
        input_text = chem_desc.input_text[i]
        
        if model_name == 'biot5+':
            input_ids = tokenizer(input_text, return_tensors = 'pt').input_ids
            with torch.no_grad():
                text_emb = model.encoder(input_ids)
                text_emb = text_emb.last_hidden_state.sum(1).squeeze(0)
        elif model_name == 'chemdfm':
            input_ids = tokenizer(input_text, return_tensors = 'pt')
            with torch.no_grad():
                text_emb = model(**input_ids, output_hidden_states = True)
                text_emb = text_emb.hidden_states[-1].sum(1).squeeze(0).cpu()

        emb_dict = {
            'input text': input_text,
            'text embedding': text_emb
        }
        chem_emb[chem_id] = emb_dict
    
    emb_none = sum(x is None for x in chem_emb.values())
    print(f'Number of failed embedding: {emb_none}')
    
    torch.save(chem_emb, f'processed/description_embedding/{model_name}_chemical_embedding')


def gene_embedding(model_name):
    tokenizer, model = load_model(model_name)
    model.eval()
    
    gene_desc = pd.read_csv('processed/description/gene_description.csv')
    gene_desc['input_text'] = 'gene id ' + gene_desc['GeneID'].astype(str) + ', ' + gene_desc['GeneName'].fillna('') + '. ' + gene_desc['SummarizedDescription'].fillna('')
    gene_desc['input_text'] = gene_desc['input_text'].apply(lambda x: x.strip())

    gene_emb = {x: None for x in gene_desc.GeneID}
    for i in tqdm(range(len(gene_desc))):
        gene_id = gene_desc.GeneID[i]
        input_text = gene_desc.input_text[i]
        
        if model_name == 'biot5+':
            input_ids = tokenizer(input_text, return_tensors = 'pt').input_ids
            with torch.no_grad():
                text_emb = model.encoder(input_ids)
                text_emb = text_emb.last_hidden_state.sum(1).squeeze(0)
        elif model_name == 'chemdfm':
            input_ids = tokenizer(input_text, return_tensors = 'pt')
            with torch.no_grad():
                text_emb = model(**input_ids, output_hidden_states = True)
                text_emb = text_emb.hidden_states[-1].sum(1).squeeze(0).cpu()
        
        text_emb = text_emb.last_hidden_state.sum(1).squeeze(0)
        
        emb_dict = {
            'input text': input_text,
            'text embedding': text_emb
        }
        gene_emb[gene_id] = emb_dict
        
        if i % 10000 == 0:
            torch.save(gene_emb, f'processed/description_embedding/{model_name}_gene_embedding')
            print(f'saved at {i}-th gene')
    
    emb_none = sum(x is None for x in gene_emb.values())
    print('')
    print(f'Number of failed embedding: {emb_none}')
    
    torch.save(gene_emb, f'processed/description_embedding/{model_name}_gene_embedding')


def disease_embedding(model_name):
    tokenizer, model = load_model(model_name)
    model.eval()
    
    dis_desc = pd.read_csv('processed/description/disease_description.csv')
    dis_desc['input_text'] = dis_desc['DiseaseName'].fillna('') + '. ' + dis_desc['SummarizedDescription'].fillna('')
    dis_desc['input_text'] = dis_desc['input_text'].apply(lambda x: x.strip())

    dis_emb = {x: None for x in dis_desc.DiseaseID}
    for i in tqdm(range(len(dis_desc))):
        dis_id = dis_desc.DiseaseID[i]
        input_text = dis_desc.input_text[i]
        
        if model_name == 'biot5+':
            input_ids = tokenizer(input_text, return_tensors = 'pt').input_ids
            with torch.no_grad():
                text_emb = model.encoder(input_ids)
                text_emb = text_emb.last_hidden_state.sum(1).squeeze(0)
        elif model_name == 'chemdfm':
            input_ids = tokenizer(input_text, return_tensors = 'pt')
            with torch.no_grad():
                text_emb = model(**input_ids, output_hidden_states = True)
                text_emb = text_emb.hidden_states[-1].sum(1).squeeze(0).cpu()
        
        text_emb = text_emb.last_hidden_state.sum(1).squeeze(0)

        emb_dict = {
            'input text': input_text,
            'text embedding': text_emb
        }
        dis_emb[dis_id] = emb_dict
    
    emb_none = sum(x is None for x in dis_emb.values())
    print(f'Number of failed embedding: {emb_none}')
    
    torch.save(dis_emb, f'processed/description_embedding/{model_name}_disease_embedding')


def phenotype_embedding(model_name):
    tokenizer, model = load_model(model_name)
    model.eval()
    
    pheno_desc = pd.read_csv('processed/description/phenotype_description.csv')
    pheno_desc['input_text'] = pheno_desc['PhenotypeName'].fillna('') + '. ' + pheno_desc['SummarizedDescription'].fillna('')
    pheno_desc['input_text'] = pheno_desc['input_text'].apply(lambda x: x.strip())

    pheno_emb = {x: None for x in pheno_desc.PhenotypeID}
    for i in tqdm(range(len(pheno_desc))):
        pheno_id = pheno_desc.PhenotypeID[i]
        input_text = pheno_desc.input_text[i]

        if model_name == 'biot5+':
            input_ids = tokenizer(input_text, return_tensors = 'pt').input_ids
            with torch.no_grad():
                text_emb = model.encoder(input_ids)
                text_emb = text_emb.last_hidden_state.sum(1).squeeze(0)
        elif model_name == 'chemdfm':
            input_ids = tokenizer(input_text, return_tensors = 'pt')
            with torch.no_grad():
                text_emb = model(**input_ids, output_hidden_states = True)
                text_emb = text_emb.hidden_states[-1].sum(1).squeeze(0).cpu()
        
        text_emb = text_emb.last_hidden_state.sum(1).squeeze(0)
        
        emb_dict = {
            'input text': input_text,
            'text embedding': text_emb
        }
        pheno_emb[pheno_id] = emb_dict
    
    emb_none = sum(x is None for x in pheno_emb.values())
    print(f'Number of failed embedding: {emb_none}')
    
    torch.save(pheno_emb, f'processed/description_embedding/{model_name}_phenotype_embedding')


def pathway_embedding(model_name):
    tokenizer, model = load_model(model_name)
    model.eval()
    
    path_desc = pd.read_csv('processed/description/pathway_description.csv')
    path_desc['input_text'] = path_desc['PathwayName'].fillna('') + '. ' + path_desc['SummarizedDescription'].fillna('')
    path_desc['input_text'] = path_desc['input_text'].apply(lambda x: x.strip())

    path_emb = {x: None for x in path_desc.PathwayID}
    for i in tqdm(range(len(path_desc))):
        path_id = path_desc.PathwayID[i]
        input_text = path_desc.input_text[i]
        
        if model_name == 'biot5+':
            input_ids = tokenizer(input_text, return_tensors = 'pt').input_ids
            with torch.no_grad():
                text_emb = model.encoder(input_ids)
                text_emb = text_emb.last_hidden_state.sum(1).squeeze(0)
        elif model_name == 'chemdfm':
            input_ids = tokenizer(input_text, return_tensors = 'pt')
            with torch.no_grad():
                text_emb = model(**input_ids, output_hidden_states = True)
                text_emb = text_emb.hidden_states[-1].sum(1).squeeze(0).cpu()

        emb_dict = {
            'input text': input_text,
            'text embedding': text_emb
        }
        path_emb[path_id] = emb_dict
    
    emb_none = sum(x is None for x in path_emb.values())
    print(f'Number of failed embedding: {emb_none}')
    
    torch.save(path_emb, f'processed/description_embedding/{model_name}_pathway_embedding')


def go_embedding(model_name):
    tokenizer, model = load_model(model_name)
    model.eval()
    
    go_desc = pd.read_csv('processed/description/go_description.csv')
    go_desc['input_text'] = go_desc['GOTermName'].fillna('') + '. ' + go_desc['SummarizedDescription'].fillna('')
    go_desc['input_text'] = go_desc['input_text'].apply(lambda x: x.strip())

    go_emb = {x: None for x in go_desc.GOTermID}
    for i in tqdm(range(len(go_desc))):
        go_id = go_desc.GOTermID[i]
        input_text = go_desc.input_text[i]
        
        if model_name == 'biot5+':
            input_ids = tokenizer(input_text, return_tensors = 'pt').input_ids
            with torch.no_grad():
                text_emb = model.encoder(input_ids)
                text_emb = text_emb.last_hidden_state.sum(1).squeeze(0)
        elif model_name == 'chemdfm':
            input_ids = tokenizer(input_text, return_tensors = 'pt')
            with torch.no_grad():
                text_emb = model(**input_ids, output_hidden_states = True)
                text_emb = text_emb.hidden_states[-1].sum(1).squeeze(0).cpu()

        emb_dict = {
            'input text': input_text,
            'text embedding': text_emb
        }
        go_emb[go_id] = emb_dict
    
    emb_none = sum(x is None for x in go_emb.values())
    print(f'Number of failed embedding: {emb_none}')
    
    torch.save(go_emb, f'processed/description_embedding/{model_name}_go_embedding')


def relation_embedding(model_name):
    tokenizer, model = load_model(model_name)
    model.eval()
    
    rels = torch.load('processed/ctd/rel_type_map')
    
    rel_names = list(rels.keys())
    rel_emb = {x: None for x in rel_names}
    for i in tqdm(range(len(rels))):
        input_text = rel_names[i]
        
        if model_name == 'biot5+':
            input_ids = tokenizer(input_text, return_tensors = 'pt').input_ids
            with torch.no_grad():
                text_emb = model.encoder(input_ids)
                text_emb = text_emb.last_hidden_state.sum(1).squeeze(0)
        elif model_name == 'chemdfm':
            input_ids = tokenizer(input_text, return_tensors = 'pt')
            with torch.no_grad():
                text_emb = model(**input_ids, output_hidden_states = True)
                text_emb = text_emb.hidden_states[-1].sum(1).squeeze(0).cpu()

        emb_dict = {
            'input text': input_text,
            'text embedding': text_emb
        }
        rel_emb[rel_names[i]] = emb_dict
    
    emb_none = sum(x is None for x in rel_emb.values())
    print(f'Number of failed embedding: {emb_none}')
    
    torch.save(rel_emb, f'processed/description_embedding/{model_name}_relation_embedding')
    

def cd_embedding_data(model_name):
    data = torch.load('processed/cd/cd.pt')
    chem_map = torch.load('processed/cd/chem_map')
    dis_map = torch.load('processed/cd/dis_map')
    rel_map = torch.load('processed/cd/rel_type_map')
    
    chem_emb = torch.load(f'processed/description_embedding/{model_name}_chemical_embedding')
    dis_emb = torch.load(f'processed/description_embedding/{model_name}_disease_embedding') 
    rel_emb = torch.load(f'processed/description_embedding/{model_name}_relation_embedding')
    
    #
    entity_dict = dict()
    cur_idx = 0
    for key in data['num_nodes_dict']:
        entity_dict[key] = (cur_idx, cur_idx + data['num_nodes_dict'][key])
        cur_idx += data['num_nodes_dict'][key]

    chem_map = {k: v + entity_dict['chemical'][0] for k, v in chem_map.items()}
    dis_map = {k: v + entity_dict['disease'][0] for k, v in dis_map.items()}

    chem_emb = {k: chem_emb[k] for k in chem_map.keys()}
    chem_emb_tensor = torch.stack([v['text embedding'] for v in chem_emb.values()])
    dis_emb = {k: dis_emb[k] for k in dis_map.keys()}
    dis_emb_tensor = torch.stack([v['text embedding'] for v in dis_emb.values()])
    
    entity_embedding = torch.cat([chem_emb_tensor, dis_emb_tensor], dim = 0)
    relation_embedding = torch.stack([rel_emb[k]['text embedding'] for k in rel_map.keys()])
    
    torch.save(entity_embedding, f'processed/cd/{model_name}_entity_embedding')
    torch.save(relation_embedding, f'processed/cd/{model_name}_relation_embedding')
    
    
def cgd_embedding_data(model_name):
    data = torch.load('processed/cgd/cgd.pt')
    chem_map = torch.load('processed/cgd/chem_map')
    gene_map = torch.load('processed/cgd/gene_map')
    dis_map = torch.load('processed/cgd/dis_map')
    rel_map = torch.load('processed/cgd/rel_type_map')
    
    chem_emb = torch.load(f'processed/description_embedding/{model_name}_chemical_embedding')
    gene_emb = torch.load(f'processed/description_embedding/{model_name}_gene_embedding')
    dis_emb = torch.load(f'processed/description_embedding/{model_name}_disease_embedding') 
    rel_emb = torch.load(f'processed/description_embedding/{model_name}_relation_embedding')
    
    #
    entity_dict = dict()
    cur_idx = 0
    for key in data['num_nodes_dict']:
        entity_dict[key] = (cur_idx, cur_idx + data['num_nodes_dict'][key])
        cur_idx += data['num_nodes_dict'][key]

    chem_map = {k: v + entity_dict['chemical'][0] for k, v in chem_map.items()}
    gene_map = {k: v + entity_dict['gene'][0] for k, v in gene_map.items()}
    dis_map = {k: v + entity_dict['disease'][0] for k, v in dis_map.items()}

    chem_emb = {k: chem_emb[k] for k in chem_map.keys()}
    chem_emb_tensor = torch.stack([v['text embedding'] for v in chem_emb.values()])
    gene_emb = {k: gene_emb[k] for k in gene_map.keys()}
    gene_emb_tensor = torch.stack([v['text embedding'] for v in gene_emb.values()])
    dis_emb = {k: dis_emb[k] for k in dis_map.keys()}
    dis_emb_tensor = torch.stack([v['text embedding'] for v in dis_emb.values()])
    
    entity_embedding = torch.cat([chem_emb_tensor, gene_emb_tensor, dis_emb_tensor], dim = 0)
    relation_embedding = torch.stack([rel_emb[k]['text embedding'] for k in rel_map.keys()])
    
    torch.save(entity_embedding, f'processed/cgd/{model_name}_entity_embedding')
    torch.save(relation_embedding, f'processed/cgd/{model_name}_relation_embedding')
    
    
def cgpd_embedding_data(model_name):
    data = torch.load('processed/cgpd/cgpd.pt')
    chem_map = torch.load('processed/cgpd/chem_map')
    gene_map = torch.load('processed/cgpd/gene_map')
    pheno_map = torch.load('processed/cgpd/pheno_map')
    dis_map = torch.load('processed/cgpd/dis_map')
    rel_map = torch.load('processed/cgpd/rel_type_map')
    
    chem_emb = torch.load(f'processed/description_embedding/{model_name}_chemical_embedding')
    gene_emb = torch.load(f'processed/description_embedding/{model_name}_gene_embedding')
    pheno_emb = torch.load(f'processed/description_embedding/{model_name}_phenotype_embedding')
    dis_emb = torch.load(f'processed/description_embedding/{model_name}_disease_embedding') 
    rel_emb = torch.load(f'processed/description_embedding/{model_name}_relation_embedding')
    
    #
    entity_dict = dict()
    cur_idx = 0
    for key in data['num_nodes_dict']:
        entity_dict[key] = (cur_idx, cur_idx + data['num_nodes_dict'][key])
        cur_idx += data['num_nodes_dict'][key]

    chem_map = {k: v + entity_dict['chemical'][0] for k, v in chem_map.items()}
    gene_map = {k: v + entity_dict['gene'][0] for k, v in gene_map.items()}
    pheno_map = {k: v + entity_dict['phenotype'][0] for k, v in pheno_map.items()}
    dis_map = {k: v + entity_dict['disease'][0] for k, v in dis_map.items()}

    chem_emb = {k: chem_emb[k] for k in chem_map.keys()}
    chem_emb_tensor = torch.stack([v['text embedding'] for v in chem_emb.values()])
    gene_emb = {k: gene_emb[k] for k in gene_map.keys()}
    gene_emb_tensor = torch.stack([v['text embedding'] for v in gene_emb.values()])
    pheno_emb = {k: pheno_emb[k] for k in pheno_map.keys()}
    pheno_emb_tensor = torch.stack([v['text embedding'] for v in pheno_emb.values()])
    dis_emb = {k: dis_emb[k] for k in dis_map.keys()}
    dis_emb_tensor = torch.stack([v['text embedding'] for v in dis_emb.values()])
    
    entity_embedding = torch.cat([chem_emb_tensor, gene_emb_tensor, pheno_emb_tensor, dis_emb_tensor], dim = 0)
    relation_embedding = torch.stack([rel_emb[k]['text embedding'] for k in rel_map.keys()])
    
    torch.save(entity_embedding, f'processed/cgpd/{model_name}_entity_embedding')
    torch.save(relation_embedding, f'processed/cgpd/{model_name}_relation_embedding')
    
    
    
def ctd_embedding_data(model_name):
    data = torch.load('processed/ctd/ctd.pt')
    chem_map = torch.load('processed/ctd/chem_map')
    gene_map = torch.load('processed/ctd/gene_map')
    pheno_map = torch.load('processed/ctd/pheno_map')
    dis_map = torch.load('processed/ctd/dis_map')
    path_map = torch.load('processed/ctd/path_map')
    go_map = torch.load('processed/ctd/go_map')
    rel_map = torch.load('processed/ctd/rel_type_map')
    
    chem_emb = torch.load(f'processed/description_embedding/{model_name}_chemical_embedding')
    gene_emb = torch.load(f'processed/description_embedding/{model_name}_gene_embedding')
    pheno_emb = torch.load(f'processed/description_embedding/{model_name}_phenotype_embedding')
    dis_emb = torch.load(f'processed/description_embedding/{model_name}_disease_embedding') 
    path_emb = torch.load(f'processed/description_embedding/{model_name}_pathway_embedding') 
    go_emb = torch.load(f'processed/description_embedding/{model_name}_go_embedding') 
    rel_emb = torch.load(f'processed/description_embedding/{model_name}_relation_embedding')
    
    #
    entity_dict = dict()
    cur_idx = 0
    for key in data['num_nodes_dict']:
        entity_dict[key] = (cur_idx, cur_idx + data['num_nodes_dict'][key])
        cur_idx += data['num_nodes_dict'][key]

    chem_map = {k: v + entity_dict['chemical'][0] for k, v in chem_map.items()}
    gene_map = {k: v + entity_dict['gene'][0] for k, v in gene_map.items()}
    pheno_map = {k: v + entity_dict['phenotype'][0] for k, v in pheno_map.items()}
    dis_map = {k: v + entity_dict['disease'][0] for k, v in dis_map.items()}
    path_map = {k: v + entity_dict['pathway'][0] for k, v in path_map.items()}
    go_map = {k: v + entity_dict['gene_ontology'][0] for k, v in go_map.items()}

    chem_emb = {k: chem_emb[k] for k in chem_map.keys()}
    chem_emb_tensor = torch.stack([v['text embedding'] for v in chem_emb.values()])
    gene_emb = {k: gene_emb[k] for k in gene_map.keys()}
    gene_emb_tensor = torch.stack([v['text embedding'] for v in gene_emb.values()])
    pheno_emb = {k: pheno_emb[k] for k in pheno_map.keys()}
    pheno_emb_tensor = torch.stack([v['text embedding'] for v in pheno_emb.values()])
    dis_emb = {k: dis_emb[k] for k in dis_map.keys()}
    dis_emb_tensor = torch.stack([v['text embedding'] for v in dis_emb.values()])
    path_emb = {k: path_emb[k] for k in path_map.keys()}
    path_emb_tensor = torch.stack([v['text embedding'] for v in path_emb.values()])
    go_emb = {k: go_emb[k] for k in go_map.keys()}
    go_emb_tensor = torch.stack([v['text embedding'] for v in go_emb.values()])
    
    entity_embedding = torch.cat([chem_emb_tensor, gene_emb_tensor, pheno_emb_tensor, dis_emb_tensor, path_emb_tensor, go_emb_tensor], dim = 0)
    relation_embedding = torch.stack([rel_emb[k]['text embedding'] for k in rel_map.keys()])
    
    torch.save(entity_embedding, f'processed/ctd/{model_name}_entity_embedding')
    torch.save(relation_embedding, f'processed/ctd/{model_name}_relation_embedding')

    
if __name__ == '__main__':
    # chemical_embedding('biot5+')
    # gene_embedding('biot5+')
    # disease_embedding('biot5+')
    # phenotype_embedding('biot5+')
    # pathway_embedding('biot5+')
    # go_embedding('biot5+')
    # relation_embedding('biot5+')
    
    relation_embedding('chemdfm')
    disease_embedding('chemdfm')
    chemical_embedding('chemdfm')
    phenotype_embedding('chemdfm')
    pathway_embedding('chemdfm')
    go_embedding('chemdfm')
    gene_embedding('chemdfm')
    
    # cd_embedding_data('biot5+')
    # cgd_embedding_data('biot5+')
    # cgpd_embedding_data('biot5+')
    # ctd_embedding_data('biot5+')
