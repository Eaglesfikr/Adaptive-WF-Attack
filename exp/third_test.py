import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 你的模型（直接 import 当前文件或复制类）
from third_my import TFModel
from WFlib.tools import data_processor

# ===== 参数 =====
dataset = "TemporalDrift"
feature = "DT"
seq_len = 5000
num_tabs = 1
batch_size = 128
device = "cuda" if torch.cuda.is_available() else "cpu"

# ===== 路径 =====
data_path = f"./datasets/{dataset}/day14.npz"
model_path = "./checkpoints3/TemporalDrift/DF/max_f1_DT.pth"

# ===== 数据 =====
X, y = data_processor.load_data(data_path, feature, seq_len, num_tabs)
print("Test:", X.shape, y.shape)

test_iter = data_processor.load_iter(X, y, batch_size, False, 0)

# ===== 模型 =====
num_classes = len(np.unique(y))

model = TFModel(num_classes=num_classes)
model.load_state_dict(torch.load(model_path, map_location=device), strict=False)

model.to(device)
model.eval()

# ===== 推理 =====
all_preds = []
all_labels = []

with torch.no_grad():
    for batch_x, batch_y in test_iter:
        batch_x = batch_x.to(device)

        outputs, _ = model(batch_x)   # TFModel 输出

        preds = torch.argmax(outputs, dim=1).cpu().numpy()

        all_preds.extend(preds)
        all_labels.extend(batch_y.numpy())

# ===== 指标 =====
acc = accuracy_score(all_labels, all_preds)
prec = precision_score(all_labels, all_preds, average='macro', zero_division=0)
rec = recall_score(all_labels, all_preds, average='macro', zero_division=0)
f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

print(f"Accuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1-score : {f1:.4f}")