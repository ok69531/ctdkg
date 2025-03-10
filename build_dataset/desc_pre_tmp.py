import os
import re
import torch
import openai
import warnings

import numpy as np
import pandas as pd

from tqdm.autonotebook import tqdm

warnings.filterwarnings('ignore')


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
            max_tokens = 150
        )

        summarized_description = response.choices[0].message.content
        
        return summarized_description
    else: return description


client = openai.OpenAI(api_key = open('openai_key', 'r').readline())

# chemical
chem_path = 'processed/description/chemical_description.csv'
raw_chem_desc = pd.read_csv(chem_path)
for i, description in enumerate(tqdm(raw_chem_desc.SummarizedDescription)):
    if (pd.notna(description)) & (str(description)[-1] != '.'):
        raw_chem_desc.at[i, 'SummarizedDescription'] = summarize_description(client, description)

        if i % 10 == 0 or i == len(raw_chem_desc) - 1:
            raw_chem_desc.to_csv(chem_path, header = True, index=False)
    else: pass


# gene
gene_path = 'processed/description/gene_description.csv'
raw_gene_desc = pd.read_csv(gene_path)
for i, description in enumerate(tqdm(raw_gene_desc.SummarizedDescription)):
    if (pd.notna(description)) & (str(description)[-1] != '.'):
        raw_gene_desc.at[i, 'SummarizedDescription'] = summarize_description(client, description)

        if i % 10 == 0 or i == len(raw_gene_desc) - 1:
            raw_gene_desc.to_csv(gene_path, header = True, index=False)
    else: pass


# disease
dis_path = 'processed/description/disease_description.csv'
raw_dis_desc = pd.read_csv(dis_path)
for i, description in enumerate(tqdm(raw_dis_desc.SummarizedDescription)):
    if (pd.notna(description)) & (str(description)[-1] != '.'):
        raw_dis_desc.at[i, 'SummarizedDescription'] = summarize_description(client, description)

        if i % 10 == 0 or i == len(raw_dis_desc) - 1:
            raw_dis_desc.to_csv(dis_path, header = True, index=False)
    else: pass


# go
go_path = 'processed/description/go_description.csv'
raw_go_desc = pd.read_csv(go_path)
for i, description in enumerate(tqdm(raw_go_desc.SummarizedDescription)):
    if (pd.notna(description)) & (str(description)[-1] != '.'):
        raw_go_desc.at[i, 'SummarizedDescription'] = summarize_description(client, description)

        if i % 10 == 0 or i == len(raw_go_desc) - 1:
            raw_go_desc.to_csv(go_path, header = True, index=False)
    else: pass


# pathway
path_path = 'processed/description/pathway_description.csv'
raw_path_desc = pd.read_csv(path_path)
for i, description in enumerate(tqdm(raw_path_desc.SummarizedDescription)):
    if (pd.notna(description)) & (str(description)[-1] != '.'):
        raw_path_desc.at[i, 'SummarizedDescription'] = summarize_description(client, description)

        if i % 10 == 0 or i == len(raw_path_desc) - 1:
            raw_path_desc.to_csv(path_path, header = True, index=False)
    else: pass


# phenotype
pheno_path = 'processed/description/phenotype_description.csv'
raw_pheno_desc = pd.read_csv(pheno_path)
for i, description in enumerate(tqdm(raw_pheno_desc.SummarizedDescription)):
    if (pd.notna(description)) & (str(description)[-1] != '.'):
        raw_pheno_desc.at[i, 'SummarizedDescription'] = summarize_description(client, description)

        if i % 10 == 0 or i == len(raw_pheno_desc) - 1:
            raw_pheno_desc.to_csv(pheno_path, header = True, index=False)
    else: pass



raw_chem_desc['FilteredDescription'].isna().sum()
raw_chem_desc['FilteredDescription'].notna().sum()

raw_gene_desc['FilteredDescription'].isna().sum()
raw_gene_desc['FilteredDescription'].notna().sum()

raw_dis_desc['FilteredDescription'].isna().sum()
raw_dis_desc['FilteredDescription'].notna().sum()

raw_pheno_desc['FilteredDescription'].isna().sum()
raw_pheno_desc['FilteredDescription'].notna().sum()

raw_path_desc['FilteredDescription'].isna().sum()
raw_path_desc['FilteredDescription'].notna().sum()

raw_go_desc['FilteredDescription'].isna().sum()
raw_go_desc['FilteredDescription'].notna().sum()