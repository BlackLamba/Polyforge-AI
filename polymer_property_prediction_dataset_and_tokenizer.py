import json

import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import re

VOCAB_SPECIAL = ['<pad>', '<bos>', '<eos>', '<unk>']


class PolymerDataset(Dataset):
    """Dataset для полимеров с SMILES и свойствами"""

    def __init__(self, df, tokenizer, property_cols, scaler=None, fit_scaler=False):
        self.tokenizer = tokenizer
        self.property_cols = property_cols
        self.smiles = df['smiles'].tolist()

        # Извлекаем свойства
        self.properties = df[property_cols].values.astype(np.float32)

        # Маска для пропущенных значений (NaN)
        self.mask = ~np.isnan(self.properties)

        # Заполняем NaN нулями для вычислений (маска исключит их из loss)
        self.properties = np.nan_to_num(self.properties, nan=0.0)

        # Нормализация
        if fit_scaler:
            self.scaler = MinMaxScaler()
            # Fit только на валидных значениях
            valid_data = self.properties.copy()
            valid_data[~self.mask] = np.nan

            # Вычисляем min/max игнорируя NaN
            self.scaler.data_min_ = np.nanmin(valid_data, axis=0)
            self.scaler.data_max_ = np.nanmax(valid_data, axis=0)
            self.scaler.data_range_ = self.scaler.data_max_ - self.scaler.data_min_
            self.scaler.data_range_[self.scaler.data_range_ == 0] = 1  # Избегаем деления на 0
            self.scaler.scale_ = 1 / self.scaler.data_range_
            self.scaler.min_ = -self.scaler.data_min_ * self.scaler.scale_
            self.scaler.n_features_in_ = len(property_cols)

            self.properties = (self.properties - self.scaler.data_min_) / self.scaler.data_range_
        elif scaler is not None:
            self.scaler = scaler
            self.properties = (self.properties - scaler.data_min_) / scaler.data_range_
        else:
            self.scaler = None

        self.properties = np.clip(self.properties, 0, 1)  # Клипуем в [0, 1]

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        smiles = self.smiles[idx]
        encoded = self.tokenizer.encode(smiles)

        return {
            'input_ids': torch.tensor(encoded, dtype=torch.long),
            'properties': torch.tensor(self.properties[idx], dtype=torch.float32),
            'mask': torch.tensor(self.mask[idx], dtype=torch.bool)
        }


class SmilesTokenizer:
    """Токенизатор для SMILES строк с поддержкой специальных токенов"""

    SMILES_REGEX = r'(\[[^\]]+\]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])'

    def __init__(self, max_len=128):
        self.max_len = max_len
        self.regex = re.compile(self.SMILES_REGEX)
        self.vocab = {}
        self.inv_vocab = {}
        self._init_special_tokens()

    def _init_special_tokens(self):
        """Инициализация специальных токенов"""
        for i, token in enumerate(VOCAB_SPECIAL):
            self.vocab[token] = i
            self.inv_vocab[i] = token

    def fit(self, smiles_list):
        """Построение словаря на основе списка SMILES"""
        all_tokens = set()
        for smiles in smiles_list:
            tokens = self.regex.findall(smiles)
            all_tokens.update(tokens)

        # Добавляем токены в словарь
        idx = len(self.vocab)
        for token in sorted(all_tokens):
            if token not in self.vocab:
                self.vocab[token] = idx
                self.inv_vocab[idx] = token
                idx += 1

        print(f"Словарь построен: {len(self.vocab)} токенов")
        return self

    def tokenize(self, smiles):
        """Токенизация SMILES строки"""
        return self.regex.findall(smiles)

    def encode(self, smiles):
        # Добавляем BOS и EOS токены
        smiles = f"<bos>{smiles}<eos>"
        tokens = self.tokenize(smiles)

        # Используем <unk> для неизвестных токенов
        ids = []
        for t in tokens:
            # Получаем ID и конвертируем в целое число
            token_id = self.vocab.get(t, self.vocab['<unk>'])
            if torch.is_tensor(token_id):
                token_id = token_id.item()
            elif isinstance(token_id, (np.ndarray, list, tuple)):
                token_id = int(token_id[0]) if len(token_id) > 0 else 0
            ids.append(int(token_id))

        # Обрезаем или дополняем до MAX_LEN
        if len(ids) > self.max_len:
            ids = ids[:self.max_len]
        else:
            pad_id = self.vocab['<pad>']
            if torch.is_tensor(pad_id):
                pad_id = pad_id.item()
            elif isinstance(pad_id, (np.ndarray, list, tuple)):
                pad_id = int(pad_id[0]) if len(pad_id) > 0 else 0
            ids += [int(pad_id)] * (self.max_len - len(ids))

        return ids

    def decode(self, ids, skip_special=True):
        """Декодирование индексов обратно в SMILES"""
        tokens = [self.inv_vocab.get(i, '') for i in ids]

        if skip_special:
            tokens = [t for t in tokens if t not in VOCAB_SPECIAL]

        return ''.join(tokens)

    def save(self, path):
        """Сохранение токенизатора"""
        with open(path, 'w') as f:
            json.dump({'vocab': self.vocab, 'max_len': self.max_len}, f)

    @classmethod
    def load(cls, path):
        """Загрузка токенизатора"""
        with open(path, 'r') as f:
            data = json.load(f)
        tokenizer = cls(max_len=data['max_len'])
        tokenizer.vocab = data['vocab']
        tokenizer.inv_vocab = {int(v): k for k, v in data['vocab'].items()}
        return tokenizer