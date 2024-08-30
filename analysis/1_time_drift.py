# python analysis/1_time_drift.py
import os
import numpy as np
import pandas as pd

for in_file in os.listdir("datasets/tmp"):
    if in_file.endswith(".npz"):
        print("########## loading...", in_file)
        data = np.load(f"datasets/tmp/{in_file}")
        X = data["X"]
        y = data["y"]
        print(f"X = {X.shape}, y = {y.shape}")