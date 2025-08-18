import torch
import torch.nn as nn


class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, dim_feedforward, output_dim):
        super(TransformerModel, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.linear = nn.Linear(d_model, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, basic_data, blood_data, text_data_encoded, raw_data, basic_risk_data_encoded, blood_risk_data_encoded):
        # 将输入数据拼接在一起
        input_data = torch.cat((
            basic_data,
            blood_data,
            text_data_encoded.view(text_data_encoded.size(0), -1),
            raw_data,
            basic_risk_data_encoded.view(basic_risk_data_encoded.size(0), -1),
            blood_risk_data_encoded.view(blood_risk_data_encoded.size(0), -1)
        ), dim=1)
        # 调整维度，添加序列长度维度，假设序列长度为 1
        input_data = input_data.unsqueeze(1)
        # 进行 Transformer 编码
        output = self.transformer_encoder(input_data)
        # 取出最后一个时间步的输出
        output = output[:, -1, :]
        # 进行回归预测
        output = self.linear(output)
        # 通过 sigmoid 函数将结果映射到 [0, 1] 区间
        output = self.sigmoid(output)
        return output
if __name__ == '__main__':
    # 示例使用代码
    # 计算输入维度
    input_dim = 6 + 7 + 768 + 13 + 768 + 768
    d_model = 2330  # 修改 d_model 为 2330 以匹配输入数据
    nhead = 10
    num_layers = 6
    dim_feedforward = 2048
    output_dim = 1  # 回归预测的输出维度为 1
    model = TransformerModel(input_dim, d_model, nhead, num_layers, dim_feedforward, output_dim)

    # 示例输入数据
    basic_data = torch.randn(32, 6)
    blood_data = torch.randn(32, 7)
    text_data_encoded = torch.randn(32, 1, 768)
    raw_data = torch.randn(32, 13)
    basic_risk_data_encoded = torch.randn(32, 1, 768)
    blood_risk_data_encoded = torch.randn(32, 1, 768)

    # 前向传播
    output = model(basic_data, blood_data, text_data_encoded, raw_data, basic_risk_data_encoded,
                   blood_risk_data_encoded)
    print(output.shape)