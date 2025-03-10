import requests
from bs4 import BeautifulSoup

import pandas as pd


go_map = {
    'GO:0008150': 'biological',
    'GO:0005575': 'cellular',
    'GO:0003674': 'molecular'
}


def download_file(url, file_name):
    response = requests.get(url)
    url_content = response.content
    
    csv_file = open(f'raw/{file_name}', 'wb')
        
    csv_file.write(url_content)
    csv_file.close()
    
    print(f'Successfully downloaded: {file_name}')
    

def download_gene_go(session, go_map):
    go_list = list(go_map.keys())

    for go_id in go_list:
        go_id_num = go_id.split(':')[-1]
        url = f'https://ctdbase.org/detail.go?type=go&acc=GO%3A{go_id_num}&view=gene'
        response = session.get(url)    
        print(f'Status Code of GO ID {go_id}: {response.status_code}')
        
        soup = BeautifulSoup(response.text, 'html.parser')
        csv_link = soup.select_one("span.export.sprite-csv_icon").find_parent("a")['href']
        
        if not csv_link.startswith('http'):
            csv_link = requests.compat.urljoin(url, csv_link)
        
        file_name = f'CTD_gene_GO_{go_map[go_id]}.csv'
        download_file(csv_link, file_name)


if __name__ == '__main__':
    session = requests.Session()
    download_gene_go(session, go_map)
