import torch
import numpy as np
from WFlib import models
from WFlib.tools import data_processor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ===== 参数 =====
dataset = "TemporalDrift"
feature = "DT"
seq_len = 5000
num_tabs = 1
batch_size = 128
device = "cuda" if torch.cuda.is_available() else "cpu"

# ===== 路径 =====
data_path = f"./datasets/{dataset}/day14.npz"
model_path = "./checkpoints/TemporalDrift/DF/max_f1_DT.pth"

# ===== 加载数据 =====
X, y = data_processor.load_data(data_path, feature, seq_len, num_tabs)
print("Test:", X.shape, y.shape)

test_iter = data_processor.load_iter(X, y, batch_size, False, 0)

# ===== 加载模型 =====
checkpoint = torch.load(model_path, map_location=device)

num_classes = checkpoint['mlp.weight'].shape[0]  # ⭐关键

model = models.DF(num_classes)
model.load_state_dict(checkpoint, strict=False)

model.to(device)
model.eval()

# ===== 测试 =====
all_preds = []
all_labels = []

with torch.no_grad():
    for batch_x, batch_y in test_iter:
        batch_x = batch_x.to(device)
        outputs, _ = model(batch_x)

        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(batch_y.numpy())

# ===== 计算指标 =====
acc = accuracy_score(all_labels, all_preds)
prec = precision_score(all_labels, all_preds, average='macro', zero_division=0)
rec = recall_score(all_labels, all_preds, average='macro', zero_division=0)
f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

print(f"Accuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1-score : {f1:.4f}")