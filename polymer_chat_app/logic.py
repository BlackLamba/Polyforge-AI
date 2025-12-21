import pandas as pd
import numpy as np

class PolymerSelector:
    def __init__(self, dataset_path):
        self.df = pd.read_csv(dataset_path)
        
        self.column_mapping = {
            'ID': ['ID', 'id'],
            'Name': ['name', 'polymer_name', 'Name'],
            'Polymer_SMILES': ['smiles', 'Polymer_SMILES'],
            'Monomer_1': ['Monomer_SMILES_1', 'monomer_1'],
            'Monomer_2': ['Monomer_SMILES_2', 'monomer_2']
        }
        
        # Определяем системные колонки, которые всегда должны быть слева
        self.system_cols = []
        for target_name, alternatives in self.column_mapping.items():
            for alt in alternatives:
                if alt in self.df.columns:
                    self.system_cols.append(alt)
                    break

    def find_best_matches(self, reqs, top_n=5):
        if not reqs:
            return pd.DataFrame()

        candidates = self.df.copy()
        candidates['similarity_score'] = 0.0
        used_props = []

        for prop, target_val in reqs.items():
            if prop in candidates.columns:
                epsilon = 1e-9
                diff = np.abs(candidates[prop] - float(target_val))
                normalized_error = diff / (np.abs(float(target_val)) + epsilon)
                candidates['similarity_score'] += normalized_error
                used_props.append(prop)

        if not used_props:
            return pd.DataFrame()

        # Находим лучшие совпадения
        best_matches = candidates.sort_values(by='similarity_score').head(top_n)
        
        # Переупорядочиваем колонки: Системные -> Запрошенные -> Остальные -> Score
        other_cols = [c for c in self.df.columns if c not in self.system_cols and c not in used_props]
        final_order = self.system_cols + used_props + other_cols + ['similarity_score']
        
        return best_matches[final_order]