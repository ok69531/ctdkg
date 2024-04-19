import os
# import logging
import requests

import pandas as pd
from tqdm.auto import tqdm

from bs4 import BeautifulSoup
from urllib.request import urlretrieve

# logging.basicConfig(format='', level=logging.INFO)

gene_df = pd.read_csv('raw/CTD_genes.csv.gz', skiprows=list(range(27))+[28], compression='gzip')
gene_ids = gene_df.GeneID

save_dir = 'raw/gene_phenotype'
if os.path.isdir(save_dir):
    pass
else:
    os.makedirs(save_dir)

passed_gene_num = 0
for i in tqdm(range(len(gene_ids))):
    try:
        gene_id = gene_ids[i]
        url = f'https://ctdbase.org/detail.go?type=gene&acc={gene_id}&view=phenotype'

        req = requests.get(url)
        soup = BeautifulSoup(req.text, 'html.parser')

        pheno_table = soup.find(id = 'phenotypeIxnTable')
        
        if 'nodatamessage' in str(pheno_table):
            passed_gene_num += 1
            # logging.info(f'GeneID {gene_id} is passed.')
            pass
        elif pheno_table == None:
            passed_gene_num += 1
            # logging.info(f'GeneID {gene_id} is passed.')
            pass
        else:
            export_link = soup.find('div', 'exportlinks')
            export_link = export_link.find_all('a', href = True)
            csv_link = [x for x in export_link if 'csv' in str(x)][0]

            urlretrieve('https://ctdbase.org'+csv_link['href'], f'{save_dir}/gene{gene_id}.csv')
    
    except requests.exceptions.ConnectTimeout:
        print('timeout', url)
        pass
