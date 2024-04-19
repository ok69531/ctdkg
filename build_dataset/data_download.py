import requests
from os import mkdir
from os.path import isfile, isdir


ctd_url_map = {
    'chemical_vocab': 'http://ctdbase.org/reports/CTD_chemicals.csv.gz',
    'gene_vocab': 'http://ctdbase.org/reports/CTD_genes.csv.gz',
    'disease_vocab': 'http://ctdbase.org/reports/CTD_diseases.csv.gz',
    'pathway_vocab': 'http://ctdbase.org/reports/CTD_pathways.csv.gz',
    'anatomy_vocab': 'http://ctdbase.org/reports/CTD_anatomy.csv.gz',
    'chem_gene_interactions': 'http://ctdbase.org/reports/CTD_chem_gene_ixns.csv.gz',
    'chem_gene_int_types': 'http://ctdbase.org/reports/CTD_chem_gene_ixn_types.csv',
    'chem_dis_associations': 'http://ctdbase.org/reports/CTD_chemicals_diseases.csv.gz',
    'chem_go_associations': 'http://ctdbase.org/reports/CTD_chem_go_enriched.csv.gz',
    'chem_pathway_associations': 'http://ctdbase.org/reports/CTD_chem_pathways_enriched.csv.gz',
    'gene_dis_associations': 'http://ctdbase.org/reports/CTD_genes_diseases.csv.gz',
    'gene_path_associations': 'http://ctdbase.org/reports/CTD_genes_pathways.csv.gz',
    'dis_path_associations': 'http://ctdbase.org/reports/CTD_diseases_pathways.csv.gz',
    'chem_pheno_interactions': 'http://ctdbase.org/reports/CTD_pheno_term_ixns.csv.gz',
    'exposure_study_associations': 'http://ctdbase.org/reports/CTD_exposure_studies.csv.gz',
    'exposure_event_associations': 'http://ctdbase.org/reports/CTD_exposure_events.csv.gz',
    'pheno_dis_biological_associations': 'http://ctdbase.org/reports/CTD_Phenotype-Disease_biological_process_associations.csv.gz',
    'pheno_dis_cellular_associations': 'http://ctdbase.org/reports/CTD_Phenotype-Disease_cellular_component_associations.csv.gz',
    'pheno_dis_molecular_associations': 'http://ctdbase.org/reports/CTD_Phenotype-Disease_molecular_function_associations.csv.gz'
}


def download_file(url, local_path, file_name):
    response = requests.get(url)
    url_content = response.content
    
    if local_path == '':
        csv_file = open(f'{file_name}', 'wb')
    elif isdir(local_path):
        csv_file = open(f'{local_path}/{file_name}', 'wb')
    else:
        mkdir(local_path)
        csv_file = open(f'{local_path}/{file_name}', 'wb')
        
    csv_file.write(url_content)
    csv_file.close()
    
    print(f'Successfully downloaded: {file_name}')


def download_ctd_data(url_map, local_path):
    '''
    Download ctd database files
    Parameters
    ----------
        url_map : dict
            source urls
        local_path : str
            local directory paty
    ----------
    '''
    print(f'>>> Downloading CTD data files')
    print('----------------------------------------------------------------------------')

    for url in url_map.values():
        file_name = url.split('/')[-1]
    
        if local_path == '':
            file_path = file_name
        else:
            file_path = f'{local_path}/{file_name}'
        
        if isfile(file_path):
            print(f'Already existed: {file_name}')
        else:
            download_file(url, local_path, file_name)
    
    print('----------------------------------------------------------------------------')


if __name__ == '__main__':
    download_ctd_data(ctd_url_map, 'raw')
