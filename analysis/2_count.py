# 统计每一类数据集的网站数量+样本数量
# python analysis/2_count.py
import os
import numpy as np
import pandas as pd

in_path = "datasets/VersionDrift"
tot = 0
for infile in os.listdir(in_path):
    if infile == "monitored_websites.npz":
        continue
    
    print(f"############## loading {infile} ##############")
    data = np.load(f"{in_path}/{infile}")
    X = data["X"]
    y = data["y"]
    num_classes = len(np.unique(y))
    assert num_classes == y.max() + 1
    print(f"shape: X={X.shape}, y={y.shape}, num_classes={num_classes}")
    tot += y.shape[0]

    total_counts = []
    for web in range(num_classes):
        total_counts.append((y==web).sum())
    total_counts = np.array(sorted(total_counts))
    print("dist of classes:", total_counts)
print("total count:", tot)