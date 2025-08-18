import numpy as np
import torch
from torch.utils.data import Dataset
from sentence_transformers import SentenceTransformer

class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.labels = data['is_coronary'].values
        self.data = self.data.drop('is_coronary', axis=1)
        self.embedding = SentenceTransformer('/home/tansty/heart/embedding/m3e-base')
        self.raw_indices = []
        self.blood_indices = []
        self.text_indices = []
        self.basic_indices = []
        self.basic_risk_indices = []
        self.blood_risk_indices = []
        for i, col in enumerate(self.data.columns):
            if col in ['gender', 'is_diabetes', 'is_hypertension', 'is_smoking', 'is_drinking',
                       'is_family_history','age','years_of_smoking','heart_rate','respiratory_rate',
                       'diastolic_pressure','systolic_pressure','body_mass_index']:
                if col in ['gender','age','is_diabetes','systolic_pressure','is_smoking']:
                    self.basic_indices.append(i)
                self.raw_indices.append(i)
            elif col.startswith('lab_biochemical_tests_'):
                if col in['lab_biochemical_tests_TC_test_result']:
                    self.basic_indices.append(i)
                self.blood_indices.append(i)
            elif col in ['concatenated_column']:
                self.text_indices.append(i)
            elif col in ['basic_risk']:
                self.basic_risk_indices.append(i)
            elif col in ['blood_risk']:
                self.blood_risk_indices.append(i)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        basic_data = self.data.iloc[idx, self.basic_indices].values.astype(np.float32)
        blood_data = self.data.iloc[idx, self.blood_indices].values.astype(np.float32)
        raw_data = self.data.iloc[idx, self.raw_indices].values.astype(np.float32)
        text_data = self.data.iloc[idx, self.text_indices].values
        basic_risk_data = self.data.iloc[idx, self.basic_risk_indices].values
        blood_risk_data = self.data.iloc[idx, self.blood_risk_indices].values
        # 使用 SentenceTransformer 模型对 text_data 进行编码
        text_data_encoded = self.embedding.encode(text_data, convert_to_tensor=True)
        basic_risk_data_encoded = self.embedding.encode(basic_risk_data, convert_to_tensor=True)
        blood_risk_data_encoded = self.embedding.encode(blood_risk_data, convert_to_tensor=True)
        labels = self.labels[idx]
        return basic_data, blood_data, text_data_encoded,raw_data,basic_risk_data_encoded,blood_risk_data_encoded,labels
