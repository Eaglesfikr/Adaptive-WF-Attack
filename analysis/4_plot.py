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
import torch.nn.functional as F
import warnings
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import seaborn as sns
import scienceplots
warnings.filterwarnings("ignore")

def run_plot(data1, data2, ax, title_name):
    #sns.histplot(data=valid_data, kde=True, linewidth=0.01, linestyle="", alpha=0.4, binwidth=10, ax=ax, color="#904579", stat="probability",line_kws={'linewidth': 2.5})
    sns.histplot(data=data1, ax=ax, kde=True, stat="probability", label="ac", cumulative=False)
    sns.histplot(data=data2, ax=ax, kde=True, stat="probability", label="wa", cumulative=False)
    ax.set_title(title_name)
    ax.legend()

def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)

def compute_energy_score(logits, T=1.0):
    """
    计算能量分数

    参数:
        logits (torch.Tensor): 模型输出的logits张量，形状为 (batch_size, num_classes)
        T (float): 温度参数，默认值为 1.0

    返回:
        energy_scores (torch.Tensor): 能量分数张量，形状为 (batch_size,)
    """
    # 计算能量分数
    energy_scores = -T * torch.logsumexp(logits / T, dim=1)
    return energy_scores

def cal_GMM2(cur_iter, df_model, ares_model, device):
    # Fusion两个模型生成一个数据
    all_energy = []
    all_confidence = []
    all_entropy = []
    all_preds = []
    all_true = []
    fusion_ratio = 0.5
    with torch.no_grad():
        df_model.eval()
        ares_model.eval()

        for index, cur_data in enumerate(cur_iter):
            cur_X, cur_y = cur_data[0].to(device), cur_data[1].to(device)
            df_outs = df_model(cur_X[...,:5000])
            ares_outs = ares_model(cur_X)
            outs = fusion_ratio * df_outs + (1-fusion_ratio) * ares_outs
            softmax_probs = torch.softmax(outs, dim=1)
            max_probs, preds = torch.max(softmax_probs, dim=1)

            all_entropy.append(softmax_entropy(outs).cpu().numpy())
            all_energy.append(compute_energy_score(outs, T=args.temp).cpu().numpy())
            all_confidence.append(max_probs.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_true.append(cur_y.cpu().numpy())

    all_entropy = np.concatenate(all_entropy).flatten()
    all_energy = np.concatenate(all_energy).flatten()
    all_confidence = np.concatenate(all_confidence).flatten()
    all_preds = np.concatenate(all_preds).flatten()
    all_true = np.concatenate(all_true).flatten()

    all_entropy = (all_entropy-all_entropy.min())/(all_entropy.max()-all_entropy.min())
    all_energy = (all_energy-all_energy.min())/(all_energy.max()-all_energy.min())
    
    gmm_entropy = all_entropy.reshape(-1, 1)
    
    # 建立gmm
    gmm = GaussianMixture(n_components=2, tol=1e-6)
    gmm.fit(gmm_entropy)
    all_gmms = gmm.predict_proba(gmm_entropy) 
    all_gmms = all_gmms[:,gmm.means_.argmin()]

    print(f"entropy: {all_entropy.shape}, range: {all_entropy.min()}~{all_entropy.max()}")
    print(f"energy: {all_energy.shape}, range: {all_energy.min()}~{all_energy.max()}")
    print(f"confidence: {all_confidence.shape}, range: {all_confidence.min()}~{all_confidence.max()}")
    print(f"gmm: {all_gmms.shape}, range: {all_gmms.min()}~{all_gmms.max()}")

    # plot
    ac_indices = (all_preds == all_true)
    wa_indices = (all_preds != all_true)
    print("# of ac:", ac_indices.sum())
    print("# of wa:", wa_indices.sum())

    # debug
    all_gmms[all_gmms<1e-6] = 0

    fig, axes = plt.subplots(
        nrows=1, ncols=2, 
        #constrained_layout=True, 
        figsize=(10, 4)
    )
    #run_plot(all_entropy[ac_indices], all_entropy[wa_indices], axes[0], "entropy")
    run_plot(all_energy[ac_indices], all_energy[wa_indices], axes[0], "energy")
    #run_plot(all_confidence[ac_indices], all_confidence[wa_indices], axes[2], "confidence")
    run_plot(all_gmms[ac_indices], all_gmms[wa_indices], axes[1], "gmm")

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
parser.add_argument("--origin_file", type=str, default="train", help="Original file")
parser.add_argument("--valid_file", type=str, default="valid", help="Valid file")
parser.add_argument("--test_file", type=str, default="test", help="Test file")
parser.add_argument("--feature", type=str, default="DIR", help="Feature type, options=[DIR, DT, DT2, TAM, TAF]")
parser.add_argument("--seq_len", type=int, default=5000, help="Input sequence length")

# Optimization parameters
parser.add_argument("--num_workers", type=int, default=10, help="Data loader num workers")
parser.add_argument("--batch_size", type=int, default=256, help="Batch size of train input data")
parser.add_argument("--temp", type=int, default=1, help="Temperature")

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
os.makedirs(os.path.join(ckp_path, args.model), exist_ok=True)
out_file = os.path.join(log_path, f"{args.result_file}.json")

# Load training and validation data
print("-----------------------------")
print(f"loading test file: ", os.path.join(in_path, f"{args.test_file}.npz"))
test_X, test_y = data_processor.load_data(os.path.join(in_path, f"{args.test_file}.npz"), args.feature, args.seq_len)
num_classes = len(np.unique(test_y))

# Make sure there are test samples for all categories
assert num_classes == test_y.max() + 1, "Labels are not continuous"

print(f"Test: X={test_X.shape}, y={test_y.shape}")
print(f"num_classes: {num_classes}")

# Load data into iterators
test_iter = data_processor.load_iter(test_X, test_y, args.batch_size, False, args.num_workers)

# Initialize model, optimizer, and loss function
df_model = eval(f"models.DF")(num_classes, args.max_num_tabs)
df_model.load_state_dict(torch.load(os.path.join(ckp_path, "DF", f"{args.load_name}.pth"), map_location="cpu"))
print("loading DF model...", os.path.join(ckp_path, "DF", f"{args.load_name}.pth"))
df_model.to(device)

ares_model = eval(f"models.ARES")(num_classes, args.max_num_tabs)
ares_model.load_state_dict(torch.load(os.path.join(ckp_path, "ARES", f"{args.load_name}.pth"), map_location="cpu"))
print("loading ARES model...", os.path.join(ckp_path, "ARES", f"{args.load_name}.pth"))
ares_model.to(device)

df_optimizer = torch.optim.Adam(df_model.parameters(), lr=1e-4)
ares_optimizer = torch.optim.Adam(ares_model.parameters(), lr=1e-4)
cal_GMM2(test_iter, df_model, ares_model, device)
plt.savefig(f'analysis/figs/{args.dataset}_{args.test_file}_{args.temp}.jpg', dpi=300)