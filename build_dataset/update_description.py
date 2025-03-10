import re
import os
import time
import torch
import warnings

from tqdm.auto import tqdm

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


def update_chemical_description():
    # previous description data
    prev_chem_desc = pd.read_csv('raw/description/chem_description.csv')
    
    # current data
    chem_map = torch.load('processed/ctd/chem_map')
    all_chems = pd.read_csv('raw/CTD_chemicals.csv.gz', skiprows=list(range(27))+[28], compression='gzip')
    all_chems.ChemicalID = all_chems.ChemicalID.apply(lambda x: x.split(':')[-1])
    all_chems.rename(columns = {'# ChemicalName': 'ChemicalName'}, inplace = True)
    
    chemicals = pd.merge(
        pd.DataFrame(chem_map.keys(), columns = ['ChemicalID']),
        all_chems[['ChemicalName', 'ChemicalID']], on = 'ChemicalID', how = 'left'
    )
    
    chemicals = pd.merge(left = chemicals, right = prev_chem_desc, how = 'left', on = ['ChemicalID', 'ChemicalName'])
    chem_wo_desc = chemicals[chemicals['WikiDescription'].isna() & chemicals['MeSHDescription'].isna() & chemicals['ChemIDplusDescription'].isna()]

    print(f'>>> Total Number of Chemicals: {len(chemicals)}')
    print(f'>>> Number of chemicals have to be updated: {len(chem_wo_desc)}')

    ### scrap from wikipedia
    print('Scrapping descriptions of chemicals from Wikipedia...')
    wiki_proj = wiki.Wikipedia('MyProjectName (CTDKG)', 'en')

    for i in tqdm(chem_wo_desc.index):
        chem_name = chemicals.ChemicalName[i]
        chem_page = wiki_proj.page(chem_name)
        
        if chem_page.exists():
            chem_desc = chem_page.summary
            chem_desc = re.sub(r'\s+', ' ', chem_desc)
            chemicals['WikiDescription'][i] = chem_desc
            chem_wo_desc['WikiDescription'][i] = chem_desc
        else: 
            pass

    print('Updated descriptioins from Wikipedia: {} / {}'.format(
        chem_wo_desc['WikiDescription'].notna().sum(), len(chem_wo_desc)
    ))
    chemicals.to_csv('raw/description/chem_description.csv', header=True, index=False)


    ### scrap from MeSH
    print('Scrapping descriptions of chemicals from MeSH Database...')
    chem_session = requests.Session()

    for i in tqdm(chem_wo_desc.index):
        chem_tmp = chemicals.ChemicalID[i]
        chem_url = 'https://meshb.nlm.nih.gov/record/ui?ui=' + chem_tmp
        
        response = chem_session.get(chem_url)
        
        soup = BeautifulSoup(response.text, 'html.parser')
        chem_table = soup.find('div', attrs = {'class': 'tab-content'})

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
            chem_wo_desc['MeSHDescription'][i] = chem_desc
            
        except: pass
        
    chem_session.close()
    chemicals.to_csv('raw/description/chem_description.csv', header=True, index=False)
    print('Updated descriptioins from MeSH Database: {} / {}'.format(
        chem_wo_desc['MeSHDescription'].notna().sum(), len(chem_wo_desc)
    ))


    ### scrap from ChemIDplus
    print('Scrapping descriptions of chemicals from ChemIDplus...')
    chem_session = requests.Session()

    option = webdriver.ChromeOptions()
    option.add_argument('window-size=1920,1080')
    chem_driver = webdriver.Chrome(options = option)
    chem_driver.implicitly_wait(3)

    for i in tqdm(chem_wo_desc.index):
        chem_tmp = chemicals.ChemicalID[i]
        chem_ctd_url = 'https://ctdbase.org/detail.go?type=chem&acc=' + chem_tmp
        
        ctd_response = chem_session.get(chem_ctd_url)
        
        ctd_soup = BeautifulSoup(ctd_response.text, 'html.parser')
        try:
            external_link_table = ctd_soup.find('dl', attrs = {'class': 'dblinks'})
            for t, item in enumerate(external_link_table.find_all('dt')):
                if item.get_text(strip = True) == 'ChemIDplus®':
                    chemid_url = external_link_table.find_all('dd')[t].find('a')['href']
            
            chem_driver.get(chemid_url)
            chem_url = chem_driver.find_element(By.XPATH, '//*[@id="featured-results"]/div/div[2]/div/div[1]/div[2]/div[1]/a').get_attribute('href')
            chem_driver.get(chem_url)
            
            soup = BeautifulSoup(chem_driver.page_source, 'html.parser')
            table = soup.find('section', attrs = {'id': 'Title-and-Summary'})
            table = table.find_all('div', attrs = {'class': 'p-2 sm:table-row pc-gray-border-t sm:border-0'})
            
            try:
                for t in table:
                    if t.find('div').text == 'Description':
                        chem_desc = t.find_all('div', attrs = {'class': 'break-words space-y-1'})
                
                chem_desc = '. '.join([x.text for x in chem_desc])
                
                chemicals['ChemIDplusDescription'][i] = chem_desc
                chem_wo_desc['ChemIDplusDescription'][i] = chem_desc
            
            except: pass
        except: pass
        
    chemicals.to_csv('chem_description.csv', header=True, index=False)

    chem_driver.close()
    chem_session.close()
    chemicals.to_csv('raw/description/chem_description.csv', header=True, index=False)
    print('Updated descriptioins from ChemIDplus: {} / {}'.format(
        chem_wo_desc['ChemIDplusDescription'].notna().sum(), len(chem_wo_desc)
    ))
    
    ### Saving Chemicals without Descriptions
    chem_desc_num = chemicals.iloc[:, 2:].notna().sum(axis = 1)
    chem_wo_desc = chemicals[chem_desc_num == 0][['ChemicalID', 'ChemicalName']]
    chem_wo_desc = chem_wo_desc.set_index('ChemicalID').to_dict()['ChemicalName']
    torch.save(chem_wo_desc, 'raw/description/chem_wo_desc')
    
    print('>>> Descriptions of chemicals are succesfully updated.')
    print('>>> Total Number of Chemicals: {}'.format(len(chemicals)))
    print('>>> Number of Chemicals without Descriptions: {}'.format(
        len(chemicals[chemicals['WikiDescription'].isna() & chemicals['MeSHDescription'].isna() & chemicals['ChemIDplusDescription'].isna()])
    ))



def update_gene_description():
    # previous description data
    prev_gene_desc = pd.read_csv('raw/description/gene_description.csv')
    prev_gene_wo_desc = torch.load('raw/description/gene_wo_desc')
    
    # current data
    gene_map = torch.load('processed/ctd/gene_map')
    all_genes = pd.read_csv('raw/CTD_genes.csv.gz', skiprows=list(range(27))+[28], compression='gzip')
    
    genes = pd.merge(
        pd.DataFrame(gene_map.keys(), columns = ['GeneID']),
        all_genes[['GeneName', 'GeneID']], on = 'GeneID', how = 'left'
    )
    
    genes = pd.merge(left = genes, right = prev_gene_desc, how = 'left', on = ['GeneID', 'GeneName'])
    gene_wo_desc = genes[genes['GenBankDescription'].isna() & genes['WikigenesDescription'].isna()]

    genes_to_be_updated = [i for i in gene_wo_desc.index if gene_wo_desc.GeneID[i] not in prev_gene_wo_desc.keys()]
    
    print(f'>>> Total Number of Genes: {len(genes)}')
    print(f'>>> Number of genes have to be updated: {len(gene_wo_desc)}')

    ### scrap from GenBank
    print('Scrapping descriptions of genes from GenBank...')
    
    gene_session = requests.Session()
    url_tmp = 'https://www.ncbi.nlm.nih.gov/gene/?term=%d'

    for i in tqdm(genes_to_be_updated):
        gene_id = genes.GeneID[i]
        
        try:
            url = url_tmp % gene_id
            
            response = gene_session.get(url)
            
            soup = BeautifulSoup(response.text, 'html.parser')
            gene_desc = soup.find('dt', text = 'Summary').find_next_sibling('dd').text
            gene_desc = re.sub('\[.*?\]', '', gene_desc).strip()
            
            genes['GenBankDescription'][i] = gene_desc
            gene_wo_desc['GenBankDescription'][i] = gene_desc
        except:
            pass

    gene_session.close()
    genes.to_csv('raw/description/gene_description.csv', header = True, index = False)
    print('Updated descriptioins from GenBank: {} / {}'.format(
        gene_wo_desc['GenBankDescription'].notna().sum(), len(gene_wo_desc)
    ))


    ### scrap from Wikigenes
    print('Scrapping descriptions of genes from Wikigenes...')
    
    gene_session = requests.Session()
    url_tmp = 'https://www.wikigenes.org/e/gene/e/%d.html'

    for i in tqdm(genes_to_be_updated):
        gene_id = genes.GeneID[i]
        
        try:
            url = url_tmp % gene_id
            
            response = gene_session.get(url)
            
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
            
            if gene_desc: 
                genes['WikigenesDescription'][i] = gene_desc
                gene_wo_desc['WikigenesDescription'][i] = gene_desc
            else: pass
            
        except:
            pass
        
    gene_session.close()
    genes.to_csv('raw/description/gene_description.csv', header = True, index = False)
    print('Updated descriptioins from Wikigenes: {} / {}'.format(
        gene_wo_desc['WikigenesDescription'].notna().sum(), len(gene_wo_desc)
    ))


    ### Saving Genes without Descriptions
    gene_desc_num = genes.iloc[:, 2:].notna().sum(axis = 1)
    gene_wo_desc = genes[gene_desc_num == 0][['GeneID', 'GeneName']]
    gene_wo_desc = gene_wo_desc.set_index('GeneID').to_dict()['GeneName']
    torch.save(gene_wo_desc, 'raw/description/gene_wo_desc')
    
    print('>>> Descriptions of genes are succesfully updated.')
    print('>>> Total Number of Genes: {}'.format(len(genes)))
    print('>>> Number of Genes without Descriptions: {}'.format(
        len(genes[genes['GenBankDescription'].isna() & genes['WikigenesDescription'].isna()])
    ))



def update_disease_description():
    # previous description data
    prev_dis_desc = pd.read_csv('raw/description/dis_description.csv')
    
    # current data
    dis_map = torch.load('processed/ctd/dis_map')
    all_dis = pd.read_csv('raw/CTD_diseases.csv.gz', skiprows=list(range(27))+[28], compression='gzip')
    all_dis.rename(columns = {'# DiseaseName': 'DiseaseName'}, inplace = True)
    
    diseases = pd.merge(
        pd.DataFrame(dis_map.keys(), columns = ['DiseaseID']),
        all_dis[['DiseaseName', 'DiseaseID']], on = 'DiseaseID', how = 'left'
    )
    
    diseases = pd.merge(left = diseases, right = prev_dis_desc, how = 'left', on = ['DiseaseID', 'DiseaseName'])
    dis_wo_desc = diseases[diseases['WikiDescription'].isna() & diseases['CTDDescription'].isna() & diseases['OMIMDescription'].isna()]

    print(f'>>> Total Number of Diseases: {len(diseases)}')
    print(f'>>> Number of diseases have to be updated: {len(dis_wo_desc)}')

    ### scrap from wikipedia
    print('Scrapping descriptions of diseases from Wikipedia...')
    wiki_proj = wiki.Wikipedia('MyProjectName (CTDKG)', 'en')

    for i in tqdm(dis_wo_desc.index):
        dis_name = diseases.DiseaseName[i]
        dis_page = wiki_proj.page(dis_name)
        
        if dis_page.exists():
            dis_desc = dis_page.summary
            dis_desc = re.sub(r'\s+', ' ', dis_desc)
            diseases['WikiDescription'][i] = dis_desc
            dis_wo_desc['WikiDescription'][i] = dis_desc
        else: 
            pass
            
    diseases.to_csv('raw/description/dis_description.csv', header=True, index=False)
    print('Updated descriptioins from Wikipedia: {} / {}'.format(
        dis_wo_desc['WikiDescription'].notna().sum(), len(dis_wo_desc)
    ))

    # scrap from CTD
    dis_session = requests.Session()

    for i in tqdm(dis_wo_desc.index):
        dis_tmp = diseases.DiseaseID[i].split(':')
        dis_db_type = dis_tmp[0]; dis_id = dis_tmp[1]
        dis_url = 'https://ctdbase.org/detail.go?type=disease&acc=' + dis_db_type + '%3A' + dis_id
        
        response = dis_session.get(dis_url)
        
        soup = BeautifulSoup(response.text, 'html.parser')
        dis_table = soup.find('table')
        dis_col_name = [x.text for x in dis_table.find_all('th')]
        
        try:
            desc_idx = [i for i in range(len(dis_col_name)) if 'Definition' in dis_col_name[i]][0]
            dis_desc = dis_table.find_all('td')[desc_idx].text
            
            diseases['CTDDescription'][i] = dis_desc
            dis_wo_desc['CTDDescription'][i] = dis_desc
        
        except: pass

    dis_session.close()
    diseases.to_csv('raw/description/dis_description.csv', header=True, index=False)
    print('Updated descriptioins from CTD: {} / {}'.format(
        dis_wo_desc['CTDDescription'].notna().sum(), len(dis_wo_desc)
    ))

    ### scrap from OMIM
    dis_session = requests.Session()

    for i in tqdm(dis_wo_desc.index):
        dis_tmp = diseases.DiseaseID[i].split(':')
        dis_db_type = dis_tmp[0]; dis_id = dis_tmp[1]
        ctd_url = 'https://ctdbase.org/detail.go?type=disease&acc=' + dis_db_type + '%3A' + dis_id
        
        ctd_response = dis_session.get(ctd_url)
        ctd_soup = BeautifulSoup(ctd_response.text, 'html.parser')
        
        try: 
            dis_table = ctd_soup.find('table', attrs = {'class': 'datatable'})
            rows = dis_table.find_all('tr')
            row_name = [x.find('th').get_text(strip = True) for x in rows]
            omim_idx = ['OMIM' in x for x in row_name].index(True)
            omim_url = rows[omim_idx].find('td').find_next('a')['href']
            
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
                dis_wo_desc['OMIMDescription'][i] = desc
            
        except: pass

    dis_session.close()
    diseases.to_csv('raw/description/dis_description.csv', header=True, index=False)
    print('Updated descriptioins from OMIM: {} / {}'.format(
        dis_wo_desc['OMIMDescription'].notna().sum(), len(dis_wo_desc)
    ))

    ###
    dis_desc_num = diseases.iloc[:, 2:].notna().sum(axis = 1)
    dis_wo_desc = diseases[dis_desc_num == 0][['DiseaseID', 'DiseaseName']]
    dis_wo_desc = dis_wo_desc.set_index('DiseaseID').to_dict()['DiseaseName']
    torch.save(dis_wo_desc, 'raw/description/dis_wo_desc')

    print('>>> Descriptions of diseases are succesfully updated.')
    print('>>> Total Number of Diseases: {}'.format(len(diseases)))
    print('>>> Number of Diseases without Descriptions: {}'.format(
        len(diseases[diseases['WikiDescription'].isna() & diseases['CTDDescription'].isna() & diseases['OMIMDescription'].isna()])
    ))



def update_phenotype_description():
    # previous description data
    prev_pheno_desc = pd.read_csv('raw/description/pheno_description.csv')
    
    # current data
    pheno_map = torch.load('processed/ctd/pheno_map')
    
    pheno_name_col = 'PhenotypeName'
    pheno_id_col = 'PhenotypeID'

    chem_pheno = pd.read_csv('raw/CTD_pheno_term_ixns.csv.gz', skiprows=list(range(27))+[28], compression='gzip')
    chem_pheno.rename(columns = {'phenotypename': pheno_name_col, 'phenotypeid': pheno_id_col}, inplace = True)
    file_path = 'raw/gene_phenotype'
    file_list = os.listdir(file_path)
    gene_pheno = pd.concat([pd.read_csv(f'{file_path}/{f}') for f in tqdm(file_list)])
    gene_pheno.rename(columns = {'Phenotype': pheno_name_col, 'Phenotype ID': pheno_id_col}, inplace = True)
    biological_pheno_dis_tmp = pd.read_csv(
        f'raw/CTD_Phenotype-Disease_biological_process_associations.csv.gz',
        skiprows = list(range(27))+[28], compression = 'gzip')
    biological_pheno_dis_tmp.rename(columns = {'# GOName': pheno_name_col, 'GOID': pheno_id_col}, inplace = True)
    cellular_pheno_dis_tmp = pd.read_csv(
        f'raw/CTD_Phenotype-Disease_cellular_component_associations.csv.gz',
        skiprows = list(range(27))+[28], compression = 'gzip')
    cellular_pheno_dis_tmp.rename(columns = {'# GOName': pheno_name_col, 'GOID': pheno_id_col}, inplace = True)
    molecular_pheno_dis_tmp = pd.read_csv(
        f'raw/CTD_Phenotype-Disease_molecular_function_associations.csv.gz',
        skiprows = list(range(27))+[28], compression = 'gzip')
    molecular_pheno_dis_tmp.rename(columns = {'# GOName': pheno_name_col, 'GOID': pheno_id_col}, inplace = True)
    
    biological_pheno_dis = biological_pheno_dis_tmp[[pheno_name_col, pheno_id_col]]
    cellular_pheno_dis = cellular_pheno_dis_tmp[[pheno_name_col, pheno_id_col]]
    molecular_pheno_dis = molecular_pheno_dis_tmp[[pheno_name_col, pheno_id_col]]
    
    dis_pheno = pd.concat([biological_pheno_dis, cellular_pheno_dis, molecular_pheno_dis])
    
    chem_pheno = chem_pheno[[pheno_name_col, pheno_id_col]].drop_duplicates()
    gene_pheno = gene_pheno[[pheno_name_col, pheno_id_col]].drop_duplicates()
    dis_pheno = dis_pheno[[pheno_name_col, pheno_id_col]].drop_duplicates()

    all_phenos = pd.concat([chem_pheno, gene_pheno, dis_pheno]).drop_duplicates()

    ### merging entity id and name
    phenotypes = pd.merge(
        pd.DataFrame(pheno_map.keys(), columns = [pheno_id_col]),
        all_phenos[[pheno_name_col, pheno_id_col]]
    )
    
    phenotypes = pd.merge(left = phenotypes, right = prev_pheno_desc, how = 'left', on = [pheno_name_col, pheno_id_col])
    pheno_wo_desc = phenotypes[phenotypes['CTDDescription'].isna()]

    print(f'>>> Total Number of Phenotypes: {len(phenotypes)}')
    print(f'>>> Number of phenotypes have to be updated: {len(pheno_wo_desc)}')

    ### scrap from CTD
    print('Scrapping descriptions of phenotypes from CTD...')
    
    pheno_session = requests.Session()
    for i in tqdm(pheno_wo_desc.index):
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
            pheno_wo_desc['CTDDescription'][i] = pheno_desc
        
        except: pass

    pheno_session.close()
    phenotypes.to_csv('raw/description/pheno_description.csv', header=True, index=False)
    print('Updated descriptioins from CTD: {} / {}'.format(
        pheno_wo_desc['CTDDescription'].notna().sum(), len(pheno_wo_desc)
    ))
    
    print('>>> Descriptions of phenotypes are succesfully updated.')
    print('>>> Total Number of Phenotypes: {}'.format(len(phenotypes)))
    print('>>> Number of Phenotypes without Descriptions: {}'.format(
        len(phenotypes[phenotypes['CTDDescription'].isna()])
    ))



def update_pathway_description():
    # previous description data
    prev_path_desc = pd.read_csv('raw/description/pathway_description.csv')
    
    # current data
    path_map = torch.load('processed/ctd/path_map')
    all_paths = pd.read_csv('raw/CTD_pathways.csv.gz', skiprows=list(range(27))+[28], compression='gzip')
    all_paths.rename(columns = {'# PathwayName': 'PathwayName'}, inplace = True)

    pathways = pd.merge(
        pd.DataFrame(path_map.keys(), columns = ['PathwayID']),
        all_paths, on = 'PathwayID' , how = 'left'
    )

    pathways = pd.merge(left = pathways, right = prev_path_desc, how = 'left', on = ['PathwayID', 'PathwayName'])
    path_wo_desc = pathways[pathways['CTDDescription'].isna()]

    print(f'>>> Total Number of Pathways: {len(pathways)}')
    print(f'>>> Number of pathways have to be updated: {len(path_wo_desc)}')

    ### scrap from CTD
    print('Scrapping descriptions of pathways from CTD...')
    
    path_session = requests.Session()
    for i in tqdm(path_wo_desc.index):
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
                path_wo_desc['CTDDescription'][i] = desc
                        
            except: pass
        else:pass

    path_session.close()
    pathways.to_csv('raw/description/pathway_description.csv', header=True, index=False)
    print('Updated descriptioins from CTD: {} / {}'.format(
        path_wo_desc['CTDDescription'].notna().sum(), len(path_wo_desc)
    ))

    ###
    path_desc_num = pathways.iloc[:, 2:].notna().sum(axis = 1)
    path_wo_desc = pathways[path_desc_num == 0][['PathwayID', 'PathwayName']]
    path_wo_desc = path_wo_desc.set_index('PathwayID').to_dict()['PathwayName']
    torch.save(path_wo_desc, 'raw/description/path_wo_desc')
    
    print('>>> Descriptions of pathways are succesfully updated.')
    print('>>> Total Number of Pathways: {}'.format(len(pathways)))
    print('>>> Number of Pathways without Descriptions: {}'.format(
        len(pathways[pathways['CTDDescription'].isna()])
    ))



def update_go_description():
    # previous description data
    prev_go_desc = pd.read_csv('raw/description/go_description.csv')
    
    # current data
    go_map = torch.load('processed/ctd/go_map')
    
    go_name_col = 'GOTermName'
    go_id_col = 'GOTermID'

    chem_go = pd.read_csv('raw/CTD_chem_go_enriched.csv.gz', skiprows=list(range(27))+[28], compression='gzip')
    gene_bio_go = pd.read_csv('raw/CTD_gene_GO_biological.csv')
    gene_bio_go.rename(columns = {n: n.replace(' ', '') for n in gene_bio_go.columns}, inplace = True)
    gene_cell_go = pd.read_csv('raw/CTD_gene_GO_cellular.csv')
    gene_cell_go.rename(columns = {n: n.replace(' ', '') for n in gene_cell_go.columns}, inplace = True)
    gene_mol_go = pd.read_csv('raw/CTD_gene_GO_molecular.csv')
    gene_mol_go.rename(columns = {n: n.replace(' ', '') for n in gene_mol_go.columns}, inplace = True)

    chem_go = chem_go[[go_name_col, go_id_col]].drop_duplicates()
    gene_bio_go = gene_bio_go[[go_name_col, go_id_col]].drop_duplicates()
    gene_cell_go = gene_cell_go[[go_name_col, go_id_col]].drop_duplicates()
    gene_mol_go = gene_mol_go[[go_name_col, go_id_col]].drop_duplicates()

    all_gos = pd.concat([chem_go, gene_bio_go, gene_cell_go, gene_mol_go])
    all_gos = all_gos.drop_duplicates()

    gos = pd.merge(
        pd.DataFrame(go_map.keys(), columns = ['GOTermID']),
        all_gos, on = 'GOTermID', how = 'left'
    )

    gos = pd.merge(left = gos, right = prev_go_desc, how = 'left', on = [go_name_col, go_id_col])
    go_wo_desc = gos[gos['CTDDescription'].isna()]

    print(f'>>> Total Number of GOs: {len(gos)}')
    print(f'>>> Number of gos have to be updated: {len(go_wo_desc)}')

    ### scrap from CTD
    print('Scrapping descriptions of gos from CTD...')
    
    go_session = requests.Session()
    for i in tqdm(go_wo_desc.index):
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
            go_wo_desc['CTDDescription'][i] = go_desc
        
        except: pass

    go_session.close()
    gos.to_csv('raw/description/go_description.csv', header=True, index=False)
    print('Updated descriptioins from CTD: {} / {}'.format(
        go_wo_desc['CTDDescription'].notna().sum(), len(go_wo_desc)
    ))

    ###
    go_desc_num = gos.iloc[:, 2:].notna().sum(axis = 1)
    go_wo_desc = gos[go_desc_num == 0][['GOTermID', 'GOTermName']]
    go_wo_desc = go_wo_desc.set_index('GOTermID').to_dict()['GOTermName']
    torch.save(go_wo_desc, 'raw/description/go_wo_desc')

    print('>>> Descriptions of gos are succesfully updated.')
    print('>>> Total Number of GOs: {}'.format(len(gos)))
    print('>>> Number of GOs without Descriptions: {}'.format(
        len(gos[gos['CTDDescription'].isna()])
    ))


if __name__ == '__main__':
    update_chemical_description()
    update_gene_description()
    update_disease_description()
    update_phenotype_description()
    update_pathway_description()
    update_go_description()
