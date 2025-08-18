import matplotlib
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from torch import nn, optim
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, accuracy_score, roc_auc_score, f1_score, recall_score, precision_score

from data_processing import preprocess_data
from dataset_weight import MyWeightDataset  # 使用 MyWeightDataset
from models.twoStage_aft_model import FusionAFTModel
from models.twoStage_mlpmixer_model import FusionMLPMixerModel
from models.twoStage_update_aft_model import FusionAFTUPModel
from models.twoStage_weight_dyt_model import FusionAFTUPDWModel
from models.twoStage_weight_model import FusionAFTUPWModel

# ==================== 1️⃣ 读取数据 ====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
matplotlib.rc("font", family='WenQuanYi Micro Hei')

# 训练集
data = pd.read_csv('data/20240205133204/train.csv')
data = data.applymap(lambda x: x.strip() if isinstance(x, str) else x)  # 清理数据
data = preprocess_data(data)
dataset = MyWeightDataset(data)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 测试集
test_data = pd.read_csv('data/20240205133204/test.csv')
test_data = test_data.applymap(lambda x: x.strip() if isinstance(x, str) else x)
test_data = preprocess_data(test_data)
test_dataset = MyWeightDataset(test_data)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# ==================== 2️⃣ 交叉验证 ====================
kf = KFold(n_splits=10, shuffle=True, random_state=42)  # 设置10折交叉验证
best_auc = 0.0
best_model_path = "best_model_auc.pth"
num_epochs = 40  # 训练轮数

# 用于记录所有 epoch 的训练 loss
epoch_losses = []
total_start_time = time.time()

for fold, (train_idx, val_idx) in enumerate(kf.split(data)):
    print(f"Fold {fold+1}/{kf.get_n_splits()}")

    # 划分训练集和验证集
    train_data = data.iloc[train_idx]
    val_data = data.iloc[val_idx]

    # 创建数据集和数据加载器
    train_dataset = MyWeightDataset(train_data)
    val_dataset = MyWeightDataset(val_data)

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # ==================== 3️⃣ 初始化模型 ====================
    # 选择其中一个模型版本进行训练（此处以 FusionAFTUPDWModel 为例）
    # model = FusionAFTUPWModel(condition_dim=7).to(device)
    model = FusionAFTUPDWModel(condition_dim=7).to(device)  # 重新初始化模型

    # 报错，就这样，使用这个模型进行讲故事
    # model = FusionAFTUPDWModel(condition_dim=7).to(device)


    criterion = nn.BCELoss()  # 二元交叉熵损失
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 记录当前 fold 最好的 AUC
    best_fold_auc = 0.0

    # ==================== 4️⃣ 训练和验证 ====================
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        all_train_labels, all_train_preds = [], []

        for batch in train_dataloader:
            (basic_data, blood_data, text_data_encoded, raw_data,
             basic_risk_data_encoded, blood_risk_data_encoded, context_input, labels) = batch  # 获取 context_input

            # 发送到 GPU
            basic_data, blood_data, text_data_encoded, raw_data, basic_risk_data_encoded, blood_risk_data_encoded, context_input, labels = (
                basic_data.to(device), blood_data.to(device), text_data_encoded.to(device),
                raw_data.to(device), basic_risk_data_encoded.to(device), blood_risk_data_encoded.to(device),
                context_input.to(device), labels.to(device).float()
            )

            optimizer.zero_grad()
            disease_pred, _ = model(basic_data, blood_data, text_data_encoded, raw_data,
                                    basic_risk_data_encoded, blood_risk_data_encoded, context_input)

            loss = criterion(disease_pred.squeeze(), labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # 记录训练集真实值和预测值
            all_train_labels.extend(labels.cpu().numpy())
            all_train_preds.extend(disease_pred.squeeze().cpu().detach().numpy())

        avg_loss = total_loss / len(train_dataloader)
        epoch_losses.append(avg_loss)
        train_auc = roc_auc_score(all_train_labels, all_train_preds)

        # ==================== 验证 ====================
        model.eval()
        all_val_labels, all_val_preds = [], []
        with torch.no_grad():
            for batch in val_dataloader:
                (basic_data, blood_data, text_data_encoded, raw_data,
                 basic_risk_data_encoded, blood_risk_data_encoded, context_input, labels) = batch

                basic_data, blood_data, text_data_encoded, raw_data, basic_risk_data_encoded, blood_risk_data_encoded, context_input = (
                    basic_data.to(device), blood_data.to(device), text_data_encoded.to(device),
                    raw_data.to(device), basic_risk_data_encoded.to(device), blood_risk_data_encoded.to(device),
                    context_input.to(device)
                )

                disease_pred, _ = model(basic_data, blood_data, text_data_encoded, raw_data,
                                        basic_risk_data_encoded, blood_risk_data_encoded, context_input)
                all_val_labels.extend(labels.numpy())
                all_val_preds.extend(disease_pred.squeeze().cpu().numpy())

        val_auc = roc_auc_score(all_val_labels, all_val_preds)
        print(f"Fold {fold+1}, Epoch [{epoch+1}/{num_epochs}]: Train Loss: {avg_loss:.4f}, Train AUC: {train_auc:.4f}, Val AUC: {val_auc:.4f}")

        if val_auc > best_fold_auc and train_auc >= 0.70:  # 可设定训练集 AUC 阈值
            best_fold_auc = val_auc
            torch.save(model.state_dict(), best_model_path)
            print(f"🎉 New best model saved for fold {fold+1} with Val AUC: {best_fold_auc:.4f}")

    # 加载当前 fold 最好的模型
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    print(f"✅ Loaded best model for fold {fold+1} with AUC: {best_fold_auc:.4f}")
    if best_fold_auc > best_auc:
        best_auc = best_fold_auc

total_training_time = time.time() - total_start_time
print(f"🎉 Final best AUC across all folds: {best_auc:.4f}")
print(f"Total training time: {total_training_time:.2f} seconds")

# ==================== 5️⃣ 测试集评估 ====================
model.eval()
all_test_labels, all_test_preds = [], []

with torch.no_grad():
    for batch in test_dataloader:
        (basic_data, blood_data, text_data_encoded, raw_data,
         basic_risk_data_encoded, blood_risk_data_encoded, context_input, labels) = batch

        basic_data, blood_data, text_data_encoded, raw_data, basic_risk_data_encoded, blood_risk_data_encoded, context_input = (
            basic_data.to(device), blood_data.to(device), text_data_encoded.to(device),
            raw_data.to(device), basic_risk_data_encoded.to(device), blood_risk_data_encoded.to(device),
            context_input.to(device)
        )

        disease_pred, _ = model(basic_data, blood_data, text_data_encoded, raw_data,
                                basic_risk_data_encoded, blood_risk_data_encoded, context_input)
        all_test_labels.extend(labels.numpy())
        all_test_preds.extend(disease_pred.squeeze().cpu().numpy())

test_auc = roc_auc_score(all_test_labels, all_test_preds)
test_mse = mean_squared_error(all_test_labels, all_test_preds)
test_acc = accuracy_score(all_test_labels, np.round(all_test_preds))
# 采用 0.7 阈值将预测概率转换为二分类标签
test_predictions = (np.array(all_test_preds) > 0.7).astype(int)
test_labels = np.array(all_test_labels)

# 计算其他指标
test_f1 = f1_score(test_labels, test_predictions)
test_recall = recall_score(test_labels, test_predictions)
test_precision = precision_score(test_labels, test_predictions)
rmse = np.sqrt(test_mse)

print(f"Test   - MSE: {test_mse:.4f}, RMSE: {rmse:.4f}, ACC: {test_acc:.4f}, AUC: {test_auc:.4f}, "
      f"F1: {test_f1:.4f}, Recall: {test_recall:.4f}, Precision: {test_precision:.4f}")

# print(f"Test   - MSE: {test_mse:.4f}, ACC: {test_acc:.4f}, AUC: {test_auc:.4f}")

# ==================== 6️⃣ 绘制 Loss 曲线 ====================
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(epoch_losses)+1), epoch_losses, marker='o', linestyle='-', color='blue')
plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.title("Training Loss Curve")
plt.grid(True)
plt.show()
