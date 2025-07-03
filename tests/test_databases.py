import requests

def get_smiles_by_cid(cid):
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/IsomericSMILES/JSON"
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Ошибка запроса для CID {cid}")
        return None
    data = response.json()
    try:
        smiles = data['PropertyTable']['Properties'][0]['IsomericSMILES']
        return smiles
    except (KeyError, IndexError):
        print(f"SMILES не найден для CID {cid}")
        return None

# Пример:
for cid in [2244, 5957, 3672]:
    print(f"CID {cid} -> SMILES: {get_smiles_by_cid(cid)}")
