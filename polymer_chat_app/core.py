import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import pickle
import json
import re
import math
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
from io import BytesIO
from PIL import Image

# ==========================================
# 1. КОНФИГУРАЦИЯ И ВСПОМОГАТЕЛЬНЫЕ КЛАССЫ
# ==========================================

VOCAB_SPECIAL = ['<pad>', '<bos>', '<eos>', '<unk>']
MAX_LEN = 128 

class SmilesTokenizer:
    """Токенизатор (скопировано из твоего ноутбука)"""
    SMILES_REGEX = r'(\[[^\]]+\]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])'

    def __init__(self, max_len=128):
        self.max_len = max_len
        self.regex = re.compile(self.SMILES_REGEX)
        self.vocab = {}
        self.inv_vocab = {}
        self._init_special_tokens()

    def _init_special_tokens(self):
        for i, token in enumerate(VOCAB_SPECIAL):
            self.vocab[token] = i
            self.inv_vocab[i] = token

    def encode(self, smiles):
        smiles = f"<bos>{smiles}<eos>"
        tokens = self.regex.findall(smiles)
        ids = []
        for t in tokens:
            token_id = self.vocab.get(t, self.vocab['<unk>'])
            ids.append(int(token_id))
        
        if len(ids) > self.max_len:
            ids = ids[:self.max_len]
        else:
            pad_id = self.vocab['<pad>']
            ids += [int(pad_id)] * (self.max_len - len(ids))
        return ids

    @classmethod
    def load(cls, path):
        with open(path, 'r') as f:
            data = json.load(f)
        tokenizer = cls(max_len=data['max_len'])
        tokenizer.vocab = data['vocab']
        tokenizer.inv_vocab = {int(v): k for k, v in data['vocab'].items()}
        return tokenizer

# ==========================================
# 2. АРХИТЕКТУРА МОДЕЛИ (Точно как в ноутбуке)
# ==========================================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class AttentionPooling(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(emb_dim, emb_dim // 2),
            nn.Tanh(),
            nn.Linear(emb_dim // 2, 1)
        )

    def forward(self, x, mask=None):
        w = self.attention(x)
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1) 
            w = w.masked_fill(~mask_expanded, -1e4) 
        weights = torch.softmax(w, dim=1)
        return torch.sum(x * weights, dim=1)

class SmilesPropertyPredictor(nn.Module):
    def __init__(self, vocab_size, num_properties, emb_dim=256, n_heads=8, ff_dim=512, num_layers=4, dropout=0.1, max_len=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.pos_encoding = PositionalEncoding(emb_dim, max_len, dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim, nhead=n_heads, dim_feedforward=ff_dim,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.ln = nn.LayerNorm(emb_dim)
        self.pooler = AttentionPooling(emb_dim)
        self.property_head = nn.Sequential(
            nn.Linear(emb_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, ff_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim // 2, num_properties)
        )

    def forward(self, input_ids, attention_mask=None):
        if attention_mask is None:
            attention_mask = (input_ids != 0)
        src_key_padding_mask = ~attention_mask
        x = self.embedding(input_ids)
        x = self.pos_encoding(x)
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        x = self.ln(x)
        x = self.pooler(x, mask=attention_mask)
        properties = self.property_head(x)
        return properties

# ==========================================
# 3. ГЛАВНЫЙ БЕКЕНД КЛАСС
# ==========================================

class PolymerBackend:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Loading backend on {self.device}...")
        
        # Пути к файлам (убедись, что они лежат там)
        MODEL_DIR = "models/smiles_property_predictor"
        DATA_FILE = "data/polymers.csv"

        try:
            # 1. Загрузка CSV для поиска
            self.df = pd.read_csv(DATA_FILE)
            print("Database loaded.")

            # 2. Загрузка чекпоинта
            checkpoint = torch.load(f"{MODEL_DIR}/model.pt", map_location=self.device)
            
            # 3. Загрузка токенизатора
            self.tokenizer = SmilesTokenizer.load(f"{MODEL_DIR}/tokenizer.json")
            
            # 4. Загрузка скейлера
            with open(f"{MODEL_DIR}/scaler.pkl", 'rb') as f:
                self.scaler = pickle.load(f)
            
            self.property_cols = checkpoint['property_cols']
            config = checkpoint['model_config']
            
            # 5. Инициализация модели
            self.model = SmilesPropertyPredictor(**config).to(self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            print("Model loaded successfully.")
            
        except Exception as e:
            print(f"CRITICAL ERROR LOADING MODEL: {e}")
            self.model = None

    def get_molecule_image(self, smiles):
        """Генерирует картинку молекулы для UI"""
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            d = rdMolDraw2D.MolDraw2DCairo(300, 300)
            d.drawOptions().addAtomIndices = False
            d.DrawMolecule(mol)
            d.FinishDrawing()
            png_data = d.GetDrawingText()
            return Image.open(BytesIO(png_data))
        return None

    def predict(self, smiles):
        """Предсказание свойств"""
        if self.model is None:
            return "Ошибка: Модель не загружена."
            
        # Валидация
        if not Chem.MolFromSmiles(smiles):
            return "Ошибка: Некорректный SMILES."

        # Подготовка данных
        seq = self.tokenizer.encode(smiles)
        input_ids = torch.tensor([seq], dtype=torch.long).to(self.device)
        
        # Инференс
        with torch.no_grad():
            preds = self.model(input_ids).cpu().numpy()
        
        # Денормализация
        try:
            real_values = self.scaler.inverse_transform(preds)[0]
        except:
            # Fallback если что-то не так со скейлером
            real_values = preds[0] * self.scaler.scale_ + self.scaler.mean_

        # Формирование ответа
        result_dict = {col: val for col, val in zip(self.property_cols, real_values)}
        return result_dict

    def find_similar(self, query):
        """Поиск по базе"""
        # Поиск по имени (регистронезависимый)
        matches = self.df[self.df['Name'].astype(str).str.contains(query, case=False, na=False)]
        
        if matches.empty:
            return None
        
        # Возвращаем топ-5
        return matches[['Name', 'Polymer_SMILES', 'Egc', 'Tg']].head(5)

# Создаем инстанс один раз при импорте
backend = PolymerBackend()