import torch
import math
from torch import nn


class FusionModel(nn.Module):
    def __init__(self):
        super(FusionModel, self).__init__()
        self.num_heads = 8
        self.embed_dim = 768
        self.seq_len = 2
        self.raw_data_dim = 13
        self.basic_data_dim = 6
        self.blood_data_dim = 7

        self.attention_model = nn.MultiheadAttention(self.embed_dim, self.num_heads, batch_first=True)
        self.raw_data_encoding_model = nn.Linear(self.raw_data_dim, self.embed_dim)
        self.basic_data_encoding_model = nn.Linear(self.basic_data_dim, self.embed_dim)
        self.blood_data_encoding_model = nn.Linear(self.blood_data_dim, self.embed_dim)
        self.output_fc = nn.Linear(self.embed_dim * self.seq_len, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, basic_data, blood_data, text_data_encoded, raw_data, basic_risk_data_encoded, blood_risk_data_encoded):
        raw_data_encoded = self.raw_data_encoding_model(raw_data.unsqueeze(1))
        basic_data_encoded = self.basic_data_encoding_model(basic_data.unsqueeze(1))
        blood_data_encoded = self.blood_data_encoding_model(blood_data.unsqueeze(1))
        x = torch.cat([
            text_data_encoded,
            raw_data_encoded,
            basic_risk_data_encoded,
            blood_risk_data_encoded
            ], dim=1)

        x = self.attention_model(x, x, x)[0]
        x = x.reshape(x.shape[0], -1)
        x = self.output_fc(x)
        x = self.sigmoid(x)

        return x


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    basic_data = torch.randn(32, 6)
    blood_data = torch.randn(32, 7)
    # torch.Size([32, 1, 768]) text_data_encoded 心电图、CT的文本解读数据的向量化数据
    text_data_encoded = torch.randn(32, 1, 768)
    # torch.Size([32, 13]) raw_data 人的身高体重、血脂信息等维度临床数据
    raw_data = torch.randn(32, 13)
    # torch.Size([32, 1, 768]) basic_risk_data_encoded 根据身高体重等维度数据得出来的预测概率数据
    basic_risk_data_encoded = torch.randn(32, 1, 768)
    # torch.Size([32, 1, 768]) blood_risk_data_encoded 根据血脂等维度数据得出来的预测概率数据
    blood_risk_data_encoded = torch.randn(32, 1, 768)

    model = FusionModel()
    out = model(basic_data, blood_data, text_data_encoded, raw_data, basic_risk_data_encoded, blood_risk_data_encoded)
    print(out.shape)
