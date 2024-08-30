# 分析NetCLR的漂移数据
# python analysis/0_awf.py
import numpy as np 
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def run_shuffle(X, y):
    # 打乱顺序
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X_shuffled = X[indices]
    y_shuffled = y[indices]
    return X_shuffled, y_shuffled

infile1 = "datasets/AWF/Drift90.npz"
infile2 = "datasets/AWF/Drift5000.npz"

data1 = np.load(infile1)
data2 = np.load(infile2)

# print(data1.files)
# print(data2.files)

values_1 = {}
values_2 = {}
## drift90.npz ##
for key in data1.files:
    values_1[key] = data1[key]

## drift5000.npz ##
for key in data2.files:
    values_2[key] = data2[key]

X_superior = values_1["X_superior"]
X_inferior = values_1["X_inferior"]
y_superior = values_1["y_superior"]
y_inferior = values_1["y_inferior"]

num_classes = 93
assert np.array_equal(np.unique(y_superior), np.unique(y_inferior))

# 初始化LabelEncoder
label_encoder = LabelEncoder()
label_encoder.fit(y_inferior)

y_inferior = label_encoder.transform(y_inferior)
y_superior = label_encoder.transform(y_superior)

X_inferior, y_inferior = run_shuffle(X_inferior, y_inferior)
X_superior, y_superior = run_shuffle(X_superior, y_superior)

print(f'y_inferior={y_inferior.shape}, X_inferior={X_inferior.shape}')
print(f"y_superior={y_superior.shape}, X_superior={X_superior.shape}")

np.savez("datasets/AWF/inferior.npz", X=X_inferior, y=y_inferior)
np.savez("datasets/AWF/superior.npz", X=X_superior, y=y_superior)