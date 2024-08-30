# python analysis/3_dataset_split.py
import numpy as np
import os
import random
import argparse
from sklearn.model_selection import train_test_split

# Set a fixed seed for reproducibility
fix_seed = 2024
random.seed(fix_seed)
np.random.seed(fix_seed)

infile = "datasets/TimeDrift/0313.npz"
file_name = infile.split("/")[-1]

dataset_path = os.path.dirname(infile)
os.makedirs(dataset_path, exist_ok=True)

# Load dataset from the specified .npz file
print("loading...", infile)
data = np.load(infile)
X = data["X"]
y = data["y"]

# Ensure labels are continuous
num_classes = len(np.unique(y))
assert num_classes == y.max() + 1, "Labels are not continuous"

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.9, random_state=fix_seed, stratify=y)
# Further split the training set into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, train_size=0.9, random_state=fix_seed, stratify=y_train)

# Print dataset information
print(f"Train: X = {X_train.shape}, y = {y_train.shape}")
print(f"Valid: X = {X_valid.shape}, y = {y_valid.shape}")
print(f"Test: X = {X_test.shape}, y = {y_test.shape}")

# Save the split datasets into separate .npz files
np.savez(os.path.join(dataset_path, "train_"+file_name), X = X_train, y = y_train)
np.savez(os.path.join(dataset_path, "valid_"+file_name), X = X_valid, y = y_valid)
np.savez(os.path.join(dataset_path, "test_"+file_name), X = X_test, y = y_test)