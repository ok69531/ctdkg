import warnings
import requests
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings('ignore')

chem_tmp = pd.read_csv('raw/CTD_chemicals.csv.gz', skiprows = list(range(27))+[28], compression = 'gzip')


def cas_to_smiles(cas_number: str) -> str:
    """
    주어진 CAS 번호(cas_number)에 대해 PubChem PUG REST API를 사용하여 SMILES를 가져옵니다.
    성공 시 SMILES 문자열을 반환하고, 실패 시 None을 반환합니다.
    """
    base_url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name"
    # CAS 번호를 '이름'처럼 인식시켜 조회 (정확히 매칭되지 않을 수도 있음)
    url = f"{base_url}/{cas_number}/property/CanonicalSMILES/JSON"

    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            # 구조: data["PropertyTable"]["Properties"][0]["CanonicalSMILES"]
            return data["PropertyTable"]["Properties"][0]["CanonicalSMILES"]
        else:
            return None
    except Exception:
        return None


chem = chem_tmp.copy()

chem.CasRN.notna().sum()
chem.CasRN.isna().sum()

chem['SMILES'] = None
for i in tqdm(range(len(chem))):
    casrn = chem.CasRN[i]
    
    if casrn == casrn:
        chem['SMILES'][i] = cas_to_smiles(casrn)
    else: pass

chem.to_csv('raw/CTD_chemicals_with_smiles.csv', header=True, index=False)
