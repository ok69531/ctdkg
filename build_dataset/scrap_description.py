#%%
import re
import os
import time
import torch
import warnings

from tqdm import tqdm

import numpy as np
import pandas as pd

import requests
from bs4 import BeautifulSoup
import wikipediaapi as wiki

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.service import Service
from selenium.common.exceptions import NoSuchElementException, WebDriverException, StaleElementReferenceException

from webdriver_manager.chrome import ChromeDriverManager

warnings.filterwarnings('ignore')


#%%
### mappings
chem_map = torch.load('../dataset/ctd/processed/chem_map')
gene_map = torch.load('../dataset/ctd/processed/gene_map')
dis_map = torch.load('../dataset/ctd/processed/dis_map')
go_map = torch.load('../dataset/ctd/processed/go_map')
path_map = torch.load('../dataset/ctd/processed/path_map')
pheno_map = torch.load('../dataset/ctd/processed/pheno_map')

### all entities
# chemicals, diseases, genes
all_chems = pd.read_csv('raw/CTD_chemicals.csv.gz', skiprows=list(range(27))+[28], compression='gzip')
all_chems.ChemicalID = all_chems.ChemicalID.apply(lambda x: x.split(':')[-1])
all_chems.rename(columns = {'# ChemicalName': 'ChemicalName'}, inplace = True)
all_dis = pd.read_csv('raw/CTD_diseases.csv.gz', skiprows=list(range(27))+[28], compression='gzip')
all_dis.rename(columns = {'# DiseaseName': 'DiseaseName'}, inplace = True)
all_genes = pd.read_csv('raw/CTD_genes.csv.gz', skiprows=list(range(27))+[28], compression='gzip')

# gene ontologies
go_name_col = 'GOTermName'
go_id_col = 'GOTermID'

chem_go = pd.read_csv('raw/CTD_chem_go_enriched.csv.gz', skiprows=list(range(27))+[28], compression='gzip')
gene_bio_go = pd.read_csv('raw/CTD_gene_GO_biological.csv')
gene_bio_go.rename(columns = {n: n.replace(' ', '') for n in gene_bio_go.columns}, inplace = True)
gene_cell_go = pd.read_csv('raw/CTD_gene_GO_cellular.csv')
gene_cell_go.rename(columns = {n: n.replace(' ', '') for n in gene_cell_go.columns}, inplace = True)
gene_mol_go = pd.read_csv('raw/CTD_gene_GO_molecular.csv')
gene_mol_go.rename(columns = {n: n.replace(' ', '') for n in gene_mol_go.columns}, inplace = True)
dis_bio_go = pd.read_csv('raw/CTD_Phenotype-Disease_biological_process_associations.csv.gz', skiprows=list(range(27))+[28], compression='gzip')
dis_bio_go.rename(columns = {'# GOName': 'GOTermName', 'GOID': 'GOTermID'}, inplace = True)
dis_cell_go = pd.read_csv('raw/CTD_Phenotype-Disease_cellular_component_associations.csv.gz', skiprows=list(range(27))+[28], compression='gzip')
dis_cell_go.rename(columns = {'# GOName': 'GOTermName', 'GOID': 'GOTermID'}, inplace = True)
dis_mol_go = pd.read_csv('raw/CTD_Phenotype-Disease_molecular_function_associations.csv.gz', skiprows=list(range(27))+[28], compression='gzip')
dis_mol_go.rename(columns = {'# GOName': 'GOTermName', 'GOID': 'GOTermID'}, inplace = True)

chem_go = chem_go[[go_name_col, go_id_col]].drop_duplicates()
gene_bio_go = gene_bio_go[[go_name_col, go_id_col]].drop_duplicates()
gene_cell_go = gene_cell_go[[go_name_col, go_id_col]].drop_duplicates()
gene_mol_go = gene_mol_go[[go_name_col, go_id_col]].drop_duplicates()
dis_bio_go = dis_bio_go[[go_name_col, go_id_col]].drop_duplicates()
dis_cell_go = dis_cell_go[[go_name_col, go_id_col]].drop_duplicates()
dis_mol_go = dis_mol_go[[go_name_col, go_id_col]].drop_duplicates()

all_gos = pd.concat([chem_go, gene_bio_go, gene_cell_go, gene_mol_go, dis_bio_go, dis_cell_go, dis_mol_go])
all_gos = all_gos.drop_duplicates()

# pathways
all_paths = pd.read_csv('raw/CTD_pathways.csv.gz', skiprows=list(range(27))+[28], compression='gzip')
all_paths.rename(columns = {'# PathwayName': 'PathwayName'}, inplace = True)

# phenotypes
pheno_name_col = 'PhenotypeName'
pheno_id_col = 'PhenotypeID'

chem_pheno = pd.read_csv('raw/CTD_pheno_term_ixns.csv.gz', skiprows=list(range(27))+[28], compression='gzip')
chem_pheno.rename(columns = {'phenotypename': 'PhenotypeName', 'phenotypeid': 'PhenotypeID'}, inplace = True)
file_path = 'raw/gene_phenotype'
file_list = os.listdir(file_path)
gene_pheno = pd.concat([pd.read_csv(f'{file_path}/{f}') for f in tqdm(file_list)])
gene_pheno.rename(columns = {'Phenotype': 'PhenotypeName', 'Phenotype ID': 'PhenotypeID'}, inplace = True)
file_path = 'raw/disease_phenotype'
file_list = os.listdir(file_path)
dis_pheno = pd.concat([pd.read_csv(f'{file_path}/{f}') for f in tqdm(file_list)])
dis_pheno.rename(columns = {'Phenotype Term Name': 'PhenotypeName', 'Phenotype Term ID': 'PhenotypeID'}, inplace = True)

chem_pheno = chem_pheno[[pheno_name_col, pheno_id_col]].drop_duplicates()
gene_pheno = gene_pheno[[pheno_name_col, pheno_id_col]].drop_duplicates()
dis_pheno = dis_pheno[[pheno_name_col, pheno_id_col]].drop_duplicates()

all_phenos = pd.concat([chem_pheno, gene_pheno, dis_pheno]).drop_duplicates()

### merging entity id and name
chemicals = pd.merge(
    pd.DataFrame(chem_map.keys(), columns = ['ChemicalID']),
    all_chems[['ChemicalName', 'ChemicalID']], on = 'ChemicalID', how = 'left'
)
genes = pd.merge(
    pd.DataFrame(gene_map.keys(), columns = ['GeneID']),
    all_genes[['GeneName', 'GeneID']], on = 'GeneID', how = 'left'
)
diseases = pd.merge(
    pd.DataFrame(dis_map.keys(), columns = ['DiseaseID']),
    all_dis[['DiseaseName', 'DiseaseID']], on = 'DiseaseID', how = 'left'
)
phenotypes = pd.merge(
    pd.DataFrame(pheno_map.keys(), columns = [pheno_id_col]),
    all_phenos[[pheno_name_col, pheno_id_col]]
)
pathways = pd.merge(
    pd.DataFrame(path_map.keys(), columns = ['PathwayID']),
    all_paths, on = 'PathwayID' , how = 'left'
)
gos = pd.merge(
    pd.DataFrame(go_map.keys(), columns = ['GOTermID']),
    all_gos, on = 'GOTermID', how = 'left'
)


#%%
''' Chemical '''
### description from wikipedia
# chemicals = pd.read_csv('raw/description/chem_description.csv')
wiki_proj = wiki.Wikipedia('MyProjectName (CTDKG)', 'en')

chemicals['WikiDescription'] = np.nan
for i in tqdm(range(len(chemicals))):
    chem_name = chemicals.ChemicalName[i]
    chem_page = wiki_proj.page(chem_name)
    
    if chem_page.exists():
        chem_desc = chem_page.summary
        chem_desc = re.sub(r'\s+', ' ', chem_desc)
        chemicals['WikiDescription'][i] = chem_desc
    else: 
        pass


chemicals['WikiDescription'].isna().sum()
# chemicals.to_csv('raw/description/chem_description.csv', header=True, index=False)


### description from MeSH
chem_session = requests.Session()

chemicals['MeSHDescription'] = np.nan
for i in tqdm(range(len(chemicals))):
    chem_tmp = chemicals.ChemicalID[i]
    chem_url = 'https://meshb.nlm.nih.gov/record/ui?ui=' + chem_tmp
    
    response = chem_session.get(chem_url)
    # response.raise_for_status()
    
    soup = BeautifulSoup(response.text, 'html.parser')
    chem_table = soup.find('div', attrs = {'class': 'tab-content'})
    # chem_col_name = [x.text for x in chem_table.find_all('dt')]

    try:
        data = {}
        items = chem_table.find_all(['dt', 'dd'])
        current_dt = None

        # dt, dd 태그를 순회
        for item in items:
            if item.name == 'dt':
                # dt 태그가 나오면 새로운 컬럼 이름으로 설정
                current_dt = item.get_text(strip=True)
                # 데이터 딕셔너리에 해당 컬럼 이름을 키로 설정, 값은 리스트로 초기화
                if current_dt not in data:
                    data[current_dt] = []
            elif item.name == 'dd':
                # dd 태그가 나오면 현재 dt에 해당하는 리스트에 값 추가
                if current_dt:
                    data[current_dt].append(item.get_text(strip=True))
        chem_desc = '. '.join(data['Note'])
        
        chemicals['MeSHDescription'][i] = chem_desc
    
    except: pass
    
    if i % 1000 == 0:
        chemicals.to_csv('raw/description/chem_description.csv', header=True, index=False)

chem_session.close()


### ChemIDplus의 description 수집
chem_session = requests.Session()

option = webdriver.ChromeOptions()
option.add_argument('window-size=1920,1080')
chem_driver = webdriver.Chrome(options = option)
chem_driver.implicitly_wait(3)


chemicals['ChemIDplusDescription'] = np.nan
for i in tqdm(range(len(chemicals))):
    chem_tmp = chemicals.ChemicalID[i]
    chem_ctd_url = 'https://ctdbase.org/detail.go?type=chem&acc=' + chem_tmp
    
    ctd_response = chem_session.get(chem_ctd_url)
    # response.raise_for_status()
    
    ctd_soup = BeautifulSoup(ctd_response.text, 'html.parser')
    try:
        external_link_table = ctd_soup.find('dl', attrs = {'class': 'dblinks'})
        for t, item in enumerate(external_link_table.find_all('dt')):
            if item.get_text(strip = True) == 'ChemIDplus®':
                chemid_url = external_link_table.find_all('dd')[t].find('a')['href']
        
        chem_driver.get(chemid_url)
        chem_url = chem_driver.find_element(By.XPATH, '//*[@id="featured-results"]/div/div[2]/div/div[1]/div[2]/div[1]/a').get_attribute('href')
        chem_driver.get(chem_url)
        # chem_url = driver.find_element(By.XPATH, '//*[@id="featured-results"]/div/div[2]/div/div[1]/div[2]/div[1]/a').click()
        
        
        soup = BeautifulSoup(chem_driver.page_source, 'html.parser')
        table = soup.find('section', attrs = {'id': 'Title-and-Summary'})
        table = table.find_all('div', attrs = {'class': 'p-2 sm:table-row pc-gray-border-t sm:border-0'})
        
        try:
            for t in table:
                if t.find('div').text == 'Description':
                    chem_desc = t.find_all('div', attrs = {'class': 'break-words space-y-1'})
            
            chem_desc = '. '.join([x.text for x in chem_desc])
            
            chemicals['ChemIDplusDescription'][i] = chem_desc
        
        except: pass
        
    except: pass
    
    if i % 1000 == 0:
        chemicals.to_csv('chem_description.csv', header=True, index=False)

chem_driver.close()
chem_session.close()

# chemicals.to_csv('raw/description/chem_description.csv', header=True, index=False)


### description 없는 데이터 dictionary로 저장
chem_desc_num = chemicals.iloc[:, 2:].notna().sum(axis = 1)
chem_wo_desc = chemicals[chem_desc_num == 0][['ChemicalID', 'ChemicalName']]
chem_wo_desc = chem_wo_desc.set_index('ChemicalID').to_dict()['ChemicalName']
torch.save(chem_wo_desc, 'raw/description/chem_wo_desc')



#%%
''' Disease '''
### description from wikipedia
wiki_proj = wiki.Wikipedia('MyProjectName (CTDKG)', 'en')

diseases['WikiDescription'] = np.nan
for i in tqdm(range(len(diseases))):
    dis_name = diseases.DiseaseName[i]
    dis_page = wiki_proj.page(dis_name)
    
    if dis_page.exists():
        dis_desc = dis_page.summary
        dis_desc = re.sub(r'\s+', ' ', dis_desc)
        diseases['WikiDescription'][i] = dis_desc
    else: 
        pass
        
diseases.to_csv('raw/description/dis_description.csv', header=True, index=False)


# description from CTD
dis_session = requests.Session()

diseases['CTDDescription'] = np.nan
for i in tqdm(range(len(diseases))):
    dis_tmp = diseases.DiseaseID[i].split(':')
    dis_db_type = dis_tmp[0]; dis_id = dis_tmp[1]
    dis_url = 'https://ctdbase.org/detail.go?type=disease&acc=' + dis_db_type + '%3A' + dis_id
    
    response = dis_session.get(dis_url)
    # response.raise_for_status()
    
    soup = BeautifulSoup(response.text, 'html.parser')
    dis_table = soup.find('table')
    dis_col_name = [x.text for x in dis_table.find_all('th')]
    
    try:
        desc_idx = [i for i in range(len(dis_col_name)) if 'Definition' in dis_col_name[i]][0]
        dis_desc = dis_table.find_all('td')[desc_idx].text
        
        diseases['CTDDescription'][i] = dis_desc
    
    except: pass
    
    if i % 1000 == 0:
        diseases.to_csv('raw/description/dis_description.csv', header=True, index=False)

diseases.to_csv('raw/description/dis_description.csv', header=True, index=False)
dis_session.close()


### description from OMIM
dis_session = requests.Session()

diseases['OMIMDescription'] = np.nan
for i in tqdm(range(len(diseases))):
    dis_tmp = diseases.DiseaseID[i].split(':')
    dis_db_type = dis_tmp[0]; dis_id = dis_tmp[1]
    ctd_url = 'https://ctdbase.org/detail.go?type=disease&acc=' + dis_db_type + '%3A' + dis_id
    
    ctd_response = dis_session.get(ctd_url)
    # response.raise_for_status()
    
    ctd_soup = BeautifulSoup(ctd_response.text, 'html.parser')
    
    try: 
        dis_table = ctd_soup.find('table', attrs = {'class': 'datatable'})
        rows = dis_table.find_all('tr')
        row_name = [x.find('th').get_text(strip = True) for x in rows]
        omim_idx = ['OMIM' in x for x in row_name].index(True)
        omim_url = rows[omim_idx].find('td').find_next('a')['href']
        
        # dis_driver.get(omim_url)
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36'
        }
        
        response = dis_session.get(omim_url, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        desc = []
        if soup.find('div', attrs = {'id': 'mimTextFold'}):
            text = soup.find('div', attrs = {'id': 'mimTextFold'}).get_text(strip = True)
            text = re.sub(r'\(.*?\)', '', text)
            text = re.sub(r'\s{2,}', ' ', text)
            desc.append(text)
        if soup.find('div', attrs = {'id': 'mimDescriptionFold'}):
            description = soup.find('div', attrs = {'id': 'mimDescriptionFold'}).get_text(strip = True)
            description = re.sub(r'\(.*?\)', '', description)
            description = re.sub(r'\s{2,}', ' ', description)
            desc.append(description)
        elif soup.find('div', attrs = {'id': 'mimClinicalFeaturesFold'}):
            cf = soup.find('div', attrs = {'id': 'mimClinicalFeaturesFold'}).get_text(strip = True)
            cf = re.sub(r'\(.*?\)', '', cf)
            cf = re.sub(r'\s{2,}', ' ', cf)
            desc.append(cf)
        
        if len(desc) == 0:
            pass
        else:
            desc = ' '.join(desc)
            diseases['OMIMDescription'][i] = desc
        
    except: pass
    
    if i % 1000 == 0:
        diseases.to_csv('raw/description/dis_description.csv', header=True, index=False)

diseases.to_csv('raw/description/dis_description.csv', header=True, index=False)
dis_session.close()


###
dis_desc_num = diseases.iloc[:, 2:].notna().sum(axis = 1)
dis_wo_desc = diseases[dis_desc_num == 0][['DiseaseID', 'DiseaseName']]
dis_wo_desc = dis_wo_desc.set_index('DiseaseID').to_dict()['DiseaseName']
torch.save(dis_wo_desc, 'raw/description/dis_wo_desc')


# %%
''' Gene '''
# gene은 https://www.ncbi.nlm.nih.gov/gene/{gene_id} 로 검색하면 summary 있음
# genes = pd.read_csv('raw/description/gene_description.csv')

# skip_genename = ['hypothetical protein', 'uncharacterized protein', 'Uncharacterized protein', 'expressed hypothetical protein']

session = requests.Session()
url_tmp = 'https://www.ncbi.nlm.nih.gov/gene/?term=%d'
# url_tmp = 'https://www.ncbi.nlm.nih.gov/gene/%d'

# genes['GenBankDescription'] = np.nan
for i in tqdm(range(24000, len(genes))):
    gene_id = genes.GeneID[i]
    gene_name = genes.GeneName[i]
    
    # if gene_name in skip_genename: pass
    # else:
    try:
        url = url_tmp % gene_id
        
        # response = requests.get(url)
        response = session.get(url)
        # response.status_code
        
        soup = BeautifulSoup(response.text, 'html.parser')
        gene_desc = soup.find('dt', text = 'Summary').find_next_sibling('dd').text
        gene_desc = re.sub('\[.*?\]', '', gene_desc).strip()
        
        genes['GenBankDescription'][i] = gene_desc
    except:
        pass

    if i % 1000 == 0:
        genes.to_csv('raw/description/gene_description.csv', header = True, index = False)

session.close()
genes.to_csv('raw/description/gene_description.csv', header = True, index = False)


###
gene_session = requests.Session()
url_tmp = 'https://www.wikigenes.org/e/gene/e/%d.html'

genes['WikigenesDescription'] = np.nan
for i in tqdm(range(len(genes))):
    gene_id = genes.GeneID[i]
    gene_name = genes.GeneName[i]
    
    # if gene_name in skip_genename: pass
    # else:
    try:
        url = url_tmp % gene_id
        
        response = gene_session.get(url)
        # response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        table = soup.find('div', attrs = {'class': 'orgMememoir_cc_content orgMememoir_cc_meta'})
        titles = table.find_all('h2')
        
        gene_desc = []
        for t in titles:
            if 'Ref' in t.get_text(strip = True):
                break
            else:
                ul = t.find_next_sibling('ul')
                if ul:
                    li_texts = [li.get_text(strip = True) for li in ul.find_all('li')]
                    li_texts = ' '.join(li_texts)
                    li_texts = re.sub(r'\[\d+\]', '', li_texts)
                    gene_desc.append(li_texts)
        gene_desc = ' '.join(gene_desc).strip()
        
        if gene_desc: genes['WikigenesDescription'][i] = gene_desc
        else: pass
        
    except:
        pass
    
    if i % 1000 == 0:
        genes.to_csv('raw/description/gene_description.csv', header = True, index = False)


genes.to_csv('raw/description/gene_description.csv', header = True, index = False)


###
gene_desc_num = genes.iloc[:, 2:].notna().sum(axis = 1)
gene_wo_desc = genes[gene_desc_num == 0][['GeneID', 'GeneName']]
gene_wo_desc = gene_wo_desc.set_index('GeneID').to_dict()['GeneName']
torch.save(gene_wo_desc, 'raw/description/gene_wo_desc')


# %%
''' Gene Ontology '''
gos['CTDDescription'] = np.nan

go_session = requests.Session()
for i in tqdm(range(len(gos))):
    go_id = gos.GOTermID[i].split(':')[-1]
    go_url = 'https://ctdbase.org/detail.go?type=go&acc=GO%3A' + go_id
    
    response = go_session.get(go_url)
    
    soup = BeautifulSoup(response.text, 'html.parser')
    go_table = soup.find('table')
    go_col_name = [x.text for x in go_table.find_all('th')]
    
    try:
        go_idx = [i for i in range(len(go_col_name)) if 'Definition' in go_col_name[i]][0]
        go_desc = go_table.find_all('td')[go_idx].text
        
        gos['CTDDescription'][i] = go_desc
    
    except: pass

    if i % 3000 == 0:
        gos.to_csv('raw/description/go_description.csv', header=True, index=False)

gos.to_csv('raw/description/go_description.csv', header=True, index=False)
go_session.close()

###
go_desc_num = gos.iloc[:, 2:].notna().sum(axis = 1)
go_wo_desc = gos[go_desc_num == 0][['GOTermID', 'GOTermID']]
go_wo_desc = go_wo_desc.set_index('GOTermID').to_dict()['GOTermID']
torch.save(go_wo_desc, 'raw/description/go_wo_desc')


#%%
''' Phenotype '''
phenotypes['CTDDescription'] = np.nan

pheno_session = requests.Session()
for i in tqdm(range(len(phenotypes))):
    pheno_id = phenotypes.PhenotypeID[i].split(':')[-1]
    pheno_url = 'https://ctdbase.org/detail.go?type=go&acc=GO%3A' + pheno_id
    
    response = pheno_session.get(pheno_url)
    
    soup = BeautifulSoup(response.text, 'html.parser')
    pheno_table = soup.find('table')
    pheno_col_name = [x.text for x in pheno_table.find_all('th')]
    
    try:
        pheno_idx = [i for i in range(len(pheno_col_name)) if 'Definition' in pheno_col_name[i]][0]
        pheno_desc = pheno_table.find_all('td')[pheno_idx].text
        
        phenotypes['CTDDescription'][i] = pheno_desc
    
    except: pass

    if i % 3000 == 0:
        phenotypes.to_csv('raw/description/pheno_description.csv', header=True, index=False)

phenotypes.to_csv('raw/description/pheno_description.csv', header=True, index=False)
pheno_session.close()

###
pheno_desc_num = phenotypes.iloc[:, 2:].notna().sum(axis = 1)
pheno_wo_desc = phenotypes[pheno_desc_num == 0][['PathwayID', 'PathwayName']]
pheno_wo_desc = pheno_wo_desc.set_index('PhenotypeID').to_dict()['PhenotypeName']
torch.save(pheno_wo_desc, 'raw/description/pheno_wo_desc')


#%%
path_session = requests.Session()

pathways['CTDDescription'] = np.nan
for i in tqdm(range(len(pathways))):
    pathway_id  = pathways.PathwayID[i]
    db_type, pathway_id = pathway_id.split(':')
    if db_type == 'KEGG':
        pathway_url = 'https://ctdbase.org/detail.go?type=pathway&acc=KEGG%3A' + pathway_id
    elif db_type == 'REACT':
        pathway_url = 'https://ctdbase.org/detail.go?type=pathway&acc=REACT%3A' + pathway_id
    
    pathurl_response = path_session.get(pathway_url)    
    
    pathurl_soup = BeautifulSoup(pathurl_response.text, 'html.parser')
    pathurl_table = pathurl_soup.find('table', attrs = {'class': 'datatable'})
    items = pathurl_soup.find_all(['td'])
    
    for item in items:
        try:
            path_desc_url = item.find('a')['href']
        except: pass
    
    path_desc_response = path_session.get(path_desc_url)
    
    if path_desc_response.status_code == 200:
        soup = BeautifulSoup(path_desc_response.text, 'html.parser')
        
        desc = np.nan
        try:
            if db_type == 'KEGG':
                table = soup.find('table', attrs = {'class': 'w1'})
                rows = table.find_all('tr')
                
                for r in rows:
                    col = r.find('th')
                    if col is not None and col.get_text(strip = True) == 'Description':
                        desc = r.find('div').get_text(strip = True)
                        break
            
            elif db_type == 'REACT':
                tmp  = soup.find_all('fieldset', attrs = {'class': 'fieldset-details'})
                
                for t in tmp:
                    if t.find('legend').get_text(strip = True) == 'General':break
                
                desc = t.find('div', attrs = {'class': 'details-summation'}).get_text(strip = True)
                desc = re.sub(r'\s*\([^()]*\d{4}[^()]*\)\s*', '', desc)
                desc = re.sub(r'\s+', ' ', desc)
            
            pathways['CTDDescription'][i] = desc
                    
        except: pass
    else:pass
    
    if i % 100 == 0:
        pathways.to_csv('raw/description/pathway_description.csv', header=True, index=False)

path_session.close()
pathways.to_csv('raw/description/pathway_description.csv', header=True, index=False)


###
path_desc_num = pathways.iloc[:, 2:].notna().sum(axis = 1)
path_wo_desc = pathways[path_desc_num == 0][['PathwayID', 'PathwayName']]
path_wo_desc = path_wo_desc.set_index('PathwayID').to_dict()['PathwayName']
torch.save(path_wo_desc, 'raw/description/path_wo_desc')
