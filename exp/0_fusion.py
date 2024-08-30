import os
import sys
import json
import time
import torch
import random
import argparse
import numpy as np
from WFlib import models
from WFlib.tools import data_processor, evaluator
import warnings
warnings.filterwarnings("ignore")

# Set a fixed seed for reproducibility
fix_seed = 2024
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

# Argument parser for command-line options, arguments, and sub-commands
parser = argparse.ArgumentParser(description="WFlib")
parser.add_argument("--dataset", type=str, required=True, default="Undefended", help="Dataset name")
parser.add_argument("--model", type=str, required=True, default="DF", help="Model name")
parser.add_argument("--device", type=str, default="cpu", help="Device, options=[cpu, cuda, cuda:x]")

# Threat model parameters
parser.add_argument("--max_num_tabs", type=int, default=1, 
                    help="Maximum number of tabs opened by users while browsing")
parser.add_argument("--scenario", type=str, default="closed-world", 
                    help="Attack scenario, options=[closed-world, open-world]")

# Input parameters
parser.add_argument("--valid_file", type=str, default="valid", help="Valid file")
parser.add_argument("--test_file", type=str, default="test", help="Test file")
parser.add_argument("--feature", type=str, default="DIR", help="Feature type, options=[DIR, DT, DT2, TAM, TAF]")
parser.add_argument("--seq_len", type=int, default=5000, help="Input sequence length")

# Optimization parameters
parser.add_argument("--num_workers", type=int, default=10, help="Data loader num workers")
parser.add_argument("--batch_size", type=int, default=256, help="Batch size of train input data")

# Output parameters
parser.add_argument("--eval_method", type=str, default="common", help="Method used in the evaluation, options=[common, kNN, holmes]")
parser.add_argument('--eval_metrics', nargs='+', required=True, type=str, 
                    help="Evaluation metrics, options=[Accuracy, Precision, Recall, F1-score, P@min, r-Precision]")
parser.add_argument("--log_path", type=str, default="./logs/", help="Log path")
parser.add_argument("--checkpoints", type=str, default="./checkpoints/", help="Location of model checkpoints")
parser.add_argument("--load_name", type=str, default="base", help="Name of the model file")
parser.add_argument("--result_file", type=str, default="result", help="File to save test results")

# Parse arguments
args = parser.parse_args()

# Ensure the specified device is available
if args.device.startswith("cuda"):
    assert torch.cuda.is_available(), f"The specified device {args.device} does not exist"
device = torch.device(args.device)

# Define paths for dataset, logs, and checkpoints
in_path = os.path.join("./datasets", args.dataset)
if not os.path.exists(in_path):
    raise FileNotFoundError(f"The dataset path does not exist: {in_path}")
log_path = os.path.join(args.log_path, args.dataset, args.model)
ckp_path = os.path.join(args.checkpoints, args.dataset)
os.makedirs(log_path, exist_ok=True)
out_file = os.path.join(log_path, f"{args.result_file}.json")

if os.path.exists(out_file):
    print(f"{out_file} has been generated.")
    sys.exit(0)

# Load training and validation data
print(f"loading test file: ", os.path.join(in_path, f"{args.test_file}.npz"))
test_X, test_y = data_processor.load_data(os.path.join(in_path, f"{args.test_file}.npz"), args.feature, args.seq_len)
num_classes = len(np.unique(test_y))

# Make sure there are test samples for all categories
assert num_classes == test_y.max() + 1, "Labels are not continuous"

# Print dataset information
print(f"Test: X={test_X.shape}, y={test_y.shape}")
print(f"num_classes: {num_classes}")

# Load data into iterators
test_iter = data_processor.load_iter(test_X, test_y, args.batch_size, False, args.num_workers)

# Initialize model, optimizer, and loss function
df_model = eval(f"models.DF")(num_classes, args.max_num_tabs)
df_model.load_state_dict(torch.load(os.path.join(ckp_path, "DF", f"{args.load_name}.pth"), map_location="cpu"))
df_model.to(device)

ares_model = eval(f"models.ARES")(num_classes, args.max_num_tabs)
ares_model.load_state_dict(torch.load(os.path.join(ckp_path, "ARES", f"{args.load_name}.pth"), map_location="cpu"))
ares_model.to(device)

# Evaluation
y_pred = []
y_true = []
fusion_ratio = 0.5
with torch.no_grad():
    df_model.eval()
    ares_model.eval()
    y_pred = []
    y_true = []

    for index, cur_data in enumerate(test_iter):
        cur_X, cur_y = cur_data[0].to(device), cur_data[1].to(device)
        df_outs = df_model(cur_X[...,:5000])
        ares_outs = ares_model(cur_X)
        outs = fusion_ratio * df_outs + (1-fusion_ratio) * ares_outs
        
        outs = torch.argsort(outs, dim=1, descending=True)[:,0]
        y_pred.append(outs.cpu().numpy())
        y_true.append(cur_y.cpu().numpy())

    y_pred = np.concatenate(y_pred).flatten()
    y_true = np.concatenate(y_true).flatten()

result = evaluator.measurement(y_true, y_pred, args.eval_metrics)
print(result)
with open(out_file, "w") as fp:
    json.dump(result, fp, indent=4)