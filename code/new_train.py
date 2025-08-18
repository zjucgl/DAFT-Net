import matplotlib
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch import nn, optim
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, accuracy_score, roc_auc_score  # 新增评价指标

from data_processing import preprocess_data
from dataset import MyDataset
from models.twoStage_aft_model import FusionAFTModel
from models.twoStage_mlpmixer_model import FusionMLPMixerModel
from models.twoStage_update_aft_model import FusionAFTUPModel

# ==================== 1️⃣ 读取数据 ====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
matplotlib.rc("font",family='WenQuanYi Micro Hei')

# 训练集
data = pd.read_csv('data/20240205133204/train.csv')
data = data.applymap(lambda x: x.strip() if isinstance(x, str) else x)  # 清理数据
data = preprocess_data(data)
dataset = MyDataset(data)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 测试集
test_data = pd.read_csv('data/20240205133204/test.csv')
test_data = test_data.applymap(lambda x: x.strip() if isinstance(x, str) else x)
test_data = preprocess_data(test_data)
test_dataset = MyDataset(test_data)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# ==================== 2️⃣ 初始化模型 ====================
# model = FusionAFTModel().to(device)
# model = FusionMLPMixerModel().to(device)
model = FusionAFTUPModel().to(device)
criterion = nn.BCELoss()  # 二元交叉熵损失
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 记录最佳 AUC
best_auc = 0.0
best_model_path = "best_model_auc.pth"

# ==================== 3️⃣ 训练 + 测试验证 ====================
num_epochs = 20  # 训练轮数

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    all_train_labels, all_train_preds = [], []

    for batch in dataloader:
        basic_data, blood_data, text_data_encoded, raw_data, basic_risk_data_encoded, blood_risk_data_encoded, labels = batch
        basic_data, blood_data, text_data_encoded, raw_data, basic_risk_data_encoded, blood_risk_data_encoded, labels = (
            basic_data.to(device), blood_data.to(device), text_data_encoded.to(device),
            raw_data.to(device), basic_risk_data_encoded.to(device), blood_risk_data_encoded.to(device),
            labels.to(device).float()
        )

        optimizer.zero_grad()
        disease_pred, _ = model(basic_data, blood_data, text_data_encoded, raw_data, basic_risk_data_encoded, blood_risk_data_encoded)

        loss = criterion(disease_pred.squeeze(), labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # 记录训练集真实值和预测值
        all_train_labels.extend(labels.cpu().numpy())
        all_train_preds.extend(disease_pred.squeeze().cpu().detach().numpy())

    # 计算训练集指标
    avg_loss = total_loss / len(dataloader)
    train_mse = mean_squared_error(all_train_labels, all_train_preds)
    train_acc = accuracy_score(all_train_labels, np.round(all_train_preds))
    train_auc = roc_auc_score(all_train_labels, all_train_preds)

    # ==================== 4️⃣ 在测试集上验证 ====================
    model.eval()
    all_test_labels, all_test_preds = [], []

    with torch.no_grad():
        for batch in test_dataloader:
            basic_data, blood_data, text_data_encoded, raw_data, basic_risk_data_encoded, blood_risk_data_encoded, labels = batch
            basic_data, blood_data, text_data_encoded, raw_data, basic_risk_data_encoded, blood_risk_data_encoded = (
                basic_data.to(device), blood_data.to(device), text_data_encoded.to(device),
                raw_data.to(device), basic_risk_data_encoded.to(device), blood_risk_data_encoded.to(device)
            )

            disease_pred, _ = model(basic_data, blood_data, text_data_encoded, raw_data, basic_risk_data_encoded, blood_risk_data_encoded)

            all_test_labels.extend(labels.numpy())
            all_test_preds.extend(disease_pred.squeeze().cpu().numpy())

    # 计算测试集指标
    test_mse = mean_squared_error(all_test_labels, all_test_preds)
    test_acc = accuracy_score(all_test_labels, np.round(all_test_preds))
    test_auc = roc_auc_score(all_test_labels, all_test_preds)

    print(f"Epoch [{epoch+1}/{num_epochs}]")
    print(f"Train  - Loss: {avg_loss:.4f}, MSE: {train_mse:.4f}, ACC: {train_acc:.4f}, AUC: {train_auc:.4f}")
    print(f"Test   - MSE: {test_mse:.4f}, ACC: {test_acc:.4f}, AUC: {test_auc:.4f}")

    # 如果测试集 AUC 更高，则保存模型
    if test_auc > best_auc:
        best_auc = test_auc
        torch.save(model.state_dict(), best_model_path)
        print(f"🎉 New best model saved with AUC: {best_auc:.4f}")

# 加载 AUC 最高的模型
model.load_state_dict(torch.load(best_model_path))
model.eval()
print(f"✅ Loaded best model with AUC: {best_auc:.4f}")

# ==================== 5️⃣ 计算整个数据集的特征重要性 ====================
all_importance_scores = []

with torch.no_grad():
    for batch in dataloader:
        basic_data, blood_data, text_data_encoded, raw_data, basic_risk_data_encoded, blood_risk_data_encoded, labels = batch
        basic_data, blood_data, text_data_encoded, raw_data, basic_risk_data_encoded, blood_risk_data_encoded = (
            basic_data.to(device), blood_data.to(device), text_data_encoded.to(device),
            raw_data.to(device), basic_risk_data_encoded.to(device), blood_risk_data_encoded.to(device)
        )

        _, importance_scores = model(basic_data, blood_data, text_data_encoded, raw_data, basic_risk_data_encoded, blood_risk_data_encoded)
        all_importance_scores.append(importance_scores.cpu().numpy())

# 计算特征重要性均值
all_importance_scores_np = np.mean(np.vstack(all_importance_scores), axis=0)

# 定义特征名称
# feature_names = ["超声心电图等", "原始数据","血脂信息", "WHO风险评估", "ASCVD风险评估"]
feature_names = ["超声心电图CT的文本诊断等",
                 'gender', 'age', 'is_diabetes', 'is_hypertension',
                 'is_smoking', 'years_of_smoking', 'is_drinking', 'is_family_history',
                 'heart_rate','respiratory_rate', 'diastolic_pressure','systolic_pressure',
                 'body_mass_index',
                 "TC","TG","HDL-C", 'LDL-C', 'LPa', 'apoAI', 'apoB']

# ==================== 6️⃣ 可视化 ====================

# (1) 绘制柱状图
plt.figure(figsize=(10, 5))
sns.barplot(x=feature_names, y=all_importance_scores_np, palette="Blues_r")
plt.xlabel("Feature Categories")
plt.ylabel("Feature Importance Score")
plt.title("Feature Importance (Attention-based)")
plt.xticks(rotation=30)
plt.show()

# (2) 绘制雷达图
def plot_radar_chart(values, labels, title="Feature Importance Radar Chart"):
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    values = values.tolist()
    values += values[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.fill(angles, values, color='blue', alpha=0.3)
    ax.plot(angles, values, color='blue', linewidth=2)
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    plt.title(title)
    plt.show()

plot_radar_chart(all_importance_scores_np, feature_names)
