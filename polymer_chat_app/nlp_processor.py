import pandas as pd
import numpy as np
import re
import json
from typing import Dict

class PolymerNLPProcessor:
    def __init__(self, dataset_path="data/polymers_with_names_predicted.csv", config_path="data/param_config.json"):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        self.df = pd.read_csv(dataset_path)
        self.stats = self._calculate_stats()
        
    def _calculate_stats(self):
        stats = {}
        for code in self.config.keys():
            if code in self.df.columns:
                vals = pd.to_numeric(self.df[code], errors='coerce').dropna()
                if not vals.empty:
                    stats[code] = {
                        'low': float(vals.quantile(0.15)),
                        'mid': float(vals.quantile(0.50)),
                        'high': float(vals.quantile(0.85))
                    }
        return stats

    def extract_requirements(self, text: str) -> Dict[str, float]:
            text = text.lower()
            extracted = {}

            # 1. ЧИСЛОВЫЕ ЗНАЧЕНИЯ (rho < 1.05 и т.д.)
            num_matches = re.findall(r"([a-zρ_0-9.]+)\s*[:=<>]*\s*([-+]?\d*\.\d+|\d+)", text)
            for p_name, p_val in num_matches:
                for code, info in self.config.items():
                    syns = [code.lower()] + [c.lower() for c in info.get('codes', [])]
                    if p_name in syns:
                        extracted[code] = float(p_val)

            text_numbers = re.findall(r"(\d+\.?\d*)", text)
            text_floats = [float(n) for n in text_numbers]

            # 2. МАРКЕРЫ (Проверка на наличие специфических слов)
            mandatory_markers = {
                "Eat": ["атом"],
                "Ei": ["иониз"],
                "Eea": ["сродст", "электр"],
                "Egc": ["групп"],
                "Egb": ["связ"],
                "Eib": ["внутр", "внутримол"],
                "CED": ["когез"],
                "epsc": ["статич", "постоян"],
                "epse": ["ггц", "ghz", "частот"],
                "perm": ["метан", "водород", "азот", "углекисл", "гелий", "газ", "ch4", "h2", "co2", "he", "n2"],
                "rho": ["плотн"],
                "Xe": ["сшив"],
                "Xc": ["кристалл"],
                "YM": ["юнг", "упруг", "эласт", "жестк"],
                "Td": ["дестру", "разлож", "деград", "термост"],
                "Tm": ["плавл"]
            }

            # 3. ОСНОВНОЙ ЦИКЛ ПОИСКА
            found_raw = {} 

            for code, info in self.config.items():
                if code in extracted: continue
                
                # Фильтр частот для epse
                if code.startswith("epse"):
                    freq_match = re.search(r"(\d+\.?\d*)", code)
                    if freq_match and float(freq_match.group(1)) not in text_floats:
                        continue

                # УМНАЯ ПРОВЕРКА МАРКЕРОВ
                # Ищем, есть ли для этого кода (или его префикса) обязательное слово
                must_contain = []
                for m_key, m_words in mandatory_markers.items():
                    if code.startswith(m_key):
                        must_contain = m_words
                        break
                
                if must_contain and not any(m in text for m in must_contain):
                    continue

                # ПОИСК ПО ИМЕНАМ
                for name in info.get('names', []):
                    name_low = name.lower()
                    # Берем корни по 4 буквы
                    roots = [w[:4] for w in name_low.split() if len(w) >= 4]
                    
                    if roots:
                        # Проверяем наличие ВСЕХ корней названия в тексте
                        found = all(root in text for root in roots)
                    else:
                        found = f" {name_low} " in f" {text} "

                    if found:
                        first_root = roots[0] if roots else name_low
                        pos = text.find(first_root)
                        context = text[max(0, pos-50) : min(len(text), pos+80)]
                        
                        level = 'mid'
                        adj_map = {
                            'high': ['высок', 'огромн', 'макс', 'превос', 'выдающ', 'больш', 'экстрем'],
                            'low': ['низк', 'маленьк', 'скромн', 'миним', 'небольш', 'очень низ'],
                            'mid': ['средн', 'стандарт', 'умерен', 'норм']
                        }
                        for lvl_key, adjs in adj_map.items():
                            if any(a in context for a in adjs):
                                level = lvl_key
                                break
                        
                        if code in self.stats:
                            found_raw[code] = self.stats[code][level]
                        break 

            # 4. ФИЛЬТР ПЕРЕКРЫТИЙ (Универсальный)
            final_params = found_raw.copy()
            found_codes = list(found_raw.keys())
            
            for i, c_a in enumerate(found_codes):
                for j, c_b in enumerate(found_codes):
                    if i == j: continue
                    n_a = self.config[c_a]['names'][0].lower()
                    n_b = self.config[c_b]['names'][0].lower()
                    
                    # Если "Плотность" входит в "Плотность когезионной энергии"
                    if n_a in n_b:
                        # Если общее слово (напр. "плотн") в тексте одно - удаляем более простой параметр
                        common_word = n_a.split()[0][:4]
                        if text.count(common_word) < 2:
                            if c_a in final_params:
                                del final_params[c_a]

            extracted.update(final_params)
            return extracted