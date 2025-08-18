import torch
from torch import nn
from models.dynamic_tanh import DynamicTanh

# ==================== 1️⃣ 动态条件加权网络 (DynamicConditionNetwork) ====================
class DynamicConditionNetwork(nn.Module):
    def __init__(self, condition_dim, feature_dim, num_heads=2):
        """
        condition_dim: 每个条件变量的维度（例如1，如果是标量）
        feature_dim: 目标输出的维度，与模型后续 embed_dim 对齐
        num_heads: 自注意力头数
        """
        super(DynamicConditionNetwork, self).__init__()
        # 将条件输入投影到 feature_dim 空间
        self.input_proj = nn.Linear(condition_dim, feature_dim)
        # 使用自注意力模块处理条件信息
        self.attention = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=num_heads, batch_first=True)
        # 输出层
        self.fc_out = nn.Linear(feature_dim, feature_dim)
        # 使用 Softmax 归一化生成的权重
        self.output_activation = nn.Softmax(dim=-1)

    def forward(self, context_input):
        """
        context_input: (batch_size, seq_len, condition_dim)
        输出: (batch_size, feature_dim)
        """
        x = self.input_proj(context_input)  # (batch_size, seq_len, feature_dim)
        attn_output, _ = self.attention(x, x, x)  # (batch_size, seq_len, feature_dim)
        # 聚合序列信息（这里采用平均池化）
        x_mean = attn_output.mean(dim=1)  # (batch_size, feature_dim)
        weight = self.fc_out(x_mean)       # (batch_size, feature_dim)
        weight = self.output_activation(weight)  # 归一化到 [0,1]
        return weight  # 动态生成的条件权重


# ==================== 2️⃣ StageBlock（阶段式特征抽象） ====================
class StageBlock(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(StageBlock, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.conv1d = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, padding=1)  # 1D 卷积
        self.norm = nn.LayerNorm(output_dim)
        self.activation = nn.SiLU()  # 使用 SiLU (SwiGLU) 作为激活函数
        self.residual = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()

    def forward(self, x):
        """
        x: (batch_size, input_dim)
        输出: (batch_size, output_dim)
        """
        residual = self.residual(x)
        x = self.fc(x)
        x = self.norm(x)
        x = self.activation(x)
        # 1D 卷积：扩展维度 -> 处理 -> 恢复原维度
        x = x.unsqueeze(1)  # (batch_size, 1, output_dim)
        x = self.conv1d(x).squeeze(1)  # (batch_size, output_dim)
        return x + residual  # 残差连接


# ==================== 3️⃣ 改进版 AFTBlock（带动态权重） ====================
class AFTBlock(nn.Module):
    def __init__(self, embed_dim):
        super(AFTBlock, self).__init__()
        self.embed_dim = embed_dim
        self.position_bias = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.fc_q = nn.Linear(embed_dim, embed_dim)
        self.fc_k = nn.Linear(embed_dim, embed_dim)
        self.fc_v = nn.Linear(embed_dim, embed_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, context_weight):
        """
        x: (batch_size, seq_len, embed_dim)
        context_weight: (batch_size, embed_dim)
        输出: (batch_size, seq_len, embed_dim)
        """
        q = self.fc_q(x)
        k = self.fc_k(x)
        v = self.fc_v(x)
        # 计算动态位置偏置
        dynamic_bias = self.position_bias * context_weight.unsqueeze(1)
        weights = self.softmax(k + dynamic_bias)
        x = weights * v  # 逐元素乘法
        return x


# ==================== 4️⃣ 主模型 (带条件自适应权重) ====================
class FusionAFTDymaicWModel(nn.Module):
    def __init__(self, condition_dim):
        super(FusionAFTDymaicWModel, self).__init__()
        self.embed_dim = 768
        self.raw_data_dim = 13
        self.basic_data_dim = 6
        self.blood_data_dim = 7

        # 使用动态条件网络（基于自注意力机制）生成动态权重
        # 假设输入的条件信息形状为 (batch_size, seq_len, condition_dim)
        self.condition_network = DynamicConditionNetwork(condition_dim, self.embed_dim, num_heads=2)

        # 阶段式特征抽象
        self.stage1 = StageBlock(self.raw_data_dim, 256)
        self.stage2 = StageBlock(256, 512)
        self.stage3 = StageBlock(512, self.embed_dim)

        # 归一化替代：使用 DynamicTanh 替代传统归一化方法，保持输入形状 (batch_size, feature_dim)
        self.raw_data_dyt = DynamicTanh(self.raw_data_dim, channels_last=True)
        self.basic_data_dyt = DynamicTanh(self.basic_data_dim, channels_last=True)
        self.blood_data_dyt = DynamicTanh(self.blood_data_dim, channels_last=True)

        # 数据编码
        self.raw_data_encoding_model = nn.Linear(self.raw_data_dim, self.embed_dim)
        self.basic_data_encoding_model = nn.Linear(self.basic_data_dim, self.embed_dim)
        self.blood_data_encoding_model = nn.Linear(self.blood_data_dim, self.embed_dim)

        # AFTBlock（带动态权重）
        self.aft_block = AFTBlock(self.embed_dim)

        # 特征融合层
        self.feature_fusion = nn.Sequential(
            nn.Linear(self.embed_dim * 5, 1024),
            nn.LayerNorm(1024),
            nn.SiLU(),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.SiLU(),
        )

        # 预测分支
        self.output_fc = nn.Linear(512, 1)
        self.sigmoid = nn.Sigmoid()

        # 解释分支（特征重要性）
        self.attn_fc = nn.Linear(512, 23)

    def forward(self, basic_data, blood_data, text_data_encoded, raw_data, basic_risk_data_encoded,
                blood_risk_data_encoded, context_input):
        """
        输入:
         - basic_data: (batch_size, 6)
         - blood_data: (batch_size, 7)
         - text_data_encoded: (batch_size, 1, embed_dim)
         - raw_data: (batch_size, 13)
         - basic_risk_data_encoded: (batch_size, 1, embed_dim)
         - blood_risk_data_encoded: (batch_size, 1, embed_dim)
         - context_input: (batch_size, seq_len, condition_dim)
        输出:
         - disease_pred: (batch_size, 1)
         - importance_scores: (batch_size, 23)
        """
        # 计算动态权重，通过动态条件网络，输出 shape: (batch_size, embed_dim)
        context_weight = self.condition_network(context_input)

        # 使用 DynamicTanh 替代传统归一化
        raw_data = self.raw_data_dyt(raw_data)       # (batch_size, 13)
        basic_data = self.basic_data_dyt(basic_data)   # (batch_size, 6)
        blood_data = self.blood_data_dyt(blood_data)   # (batch_size, 7)

        # 阶段式特征抽象
        raw_data = self.stage1(raw_data)              # (batch_size, 256)
        raw_data = self.stage2(raw_data)              # (batch_size, 512)
        raw_data_encoded = self.stage3(raw_data).unsqueeze(1)  # (batch_size, 1, embed_dim)

        # 数据编码：将基本数据和血液数据编码到 embed_dim 空间
        basic_data_encoded = self.basic_data_encoding_model(basic_data.unsqueeze(1))  # (batch_size, 1, embed_dim)
        blood_data_encoded = self.blood_data_encoding_model(blood_data.unsqueeze(1))  # (batch_size, 1, embed_dim)

        # 拼接多个模态特征：文本、原始（StageBlock 提取后）、基本数据编码、血液数据编码、风险评估编码
        x = torch.cat([text_data_encoded,
                       raw_data_encoded,
                       blood_data_encoded,
                       basic_risk_data_encoded,
                       blood_risk_data_encoded], dim=1)  # (batch_size, 5, embed_dim)

        # 经过 AFTBlock（带动态权重），输出保持 (batch_size, 5, embed_dim)
        x = self.aft_block(x, context_weight)

        # 将 5 个 embed_dim 拼接为一个长向量 (batch_size, 5 * embed_dim)
        x = x.reshape(x.shape[0], -1)  # (batch_size, 3840)

        # 经过特征融合层，将特征映射到 (batch_size, 512)
        x = self.feature_fusion(x)

        # 预测分支：输出疾病风险预测 (batch_size, 1)
        disease_pred = self.sigmoid(self.output_fc(x))

        # 解释分支：输出特征重要性 (batch_size, 23)
        importance_scores = self.attn_fc(x)

        return disease_pred, importance_scores
