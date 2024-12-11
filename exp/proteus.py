import os
import sys
import json
import time
import torch
import random
import argparse
import numpy as np
from WFlib import models
from sklearn.mixture import GaussianMixture
from WFlib.tools import data_processor, evaluator
from torch.utils.data import DataLoader
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore")

# 自适应高斯核函数
def adaptive_gaussian_kernel(source, target):
    n_samples = int(source.size(0)) + int(target.size(0))
    total = torch.cat([source, target], dim=0)
    L2_distance = ((total.unsqueeze(0) - total.unsqueeze(1)) ** 2).sum(2)
    
    # 计算带宽（根据数据自适应调整）
    bandwidth = torch.sum(L2_distance) / (n_samples ** 2 - n_samples)
    bandwidth = torch.clamp(bandwidth, min=1e-5)  # 避免数值不稳定
    
    kernel_val = torch.exp(-L2_distance / (bandwidth + 1e-5))
    return kernel_val

# 改进后的 MMD 损失函数
def cal_mmd_loss(source_features, target_features):
    batch_size = min(source_features.size(0), target_features.size(0))
    source_features = source_features[:batch_size]
    target_features = target_features[:batch_size]

    # 使用自适应高斯核
    kernels = adaptive_gaussian_kernel(source_features, target_features)

    # 计算 MMD 损失
    XX = kernels[:batch_size, :batch_size]  # 源域到源域
    YY = kernels[batch_size:, batch_size:]  # 目标域到目标域
    XY = kernels[:batch_size, batch_size:]  # 源域到目标域
    YX = kernels[batch_size:, :batch_size]  # 目标域到源域

    # 计算最终损失
    loss = torch.mean(XX + YY - XY - YX)
    return loss


def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)

def model_eval(model, test_iter, eval_metrics, device):
    with torch.no_grad():
        model.eval()
        y_pred = []
        y_true = []

        for index, cur_data in enumerate(test_iter):
            cur_X, cur_y = cur_data[0].to(device), cur_data[1].to(device)
            outs, _ = model(cur_X)
            
            cur_pred = torch.argsort(outs, dim=1, descending=True)[:,0]

            y_pred.append(cur_pred.cpu().numpy())
            y_true.append(cur_y.cpu().numpy())

        y_pred = np.concatenate(y_pred)
        y_true = np.concatenate(y_true)

    result = evaluator.measurement(y_true, y_pred, eval_metrics)
    return result

def cal_GMM_probs(model, test_iter, device):
    all_entropy = []
    all_preds = []
    with torch.no_grad():
        model.eval()
        for index, cur_data in enumerate(test_iter):
            cur_X, cur_y = cur_data[0].to(device), cur_data[1].to(device)
            outs, _ = model(cur_X)
            preds = torch.argsort(outs, dim=1, descending=True)[:,0]
            
            cur_entropy = softmax_entropy(outs)
            all_entropy.append(cur_entropy.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            
    all_entropy = np.concatenate(all_entropy).flatten()
    all_preds = np.concatenate(all_preds).flatten()

    # 归一化
    all_entropy = (all_entropy-all_entropy.min())/(all_entropy.max()-all_entropy.min())
    all_entropy = all_entropy.reshape(-1, 1)
    all_preds = torch.tensor(all_preds, dtype=torch.int64)

    # 建立gmm
    gmm = GaussianMixture(n_components=2, tol=1e-6)
    gmm.fit(all_entropy)
    prob = gmm.predict_proba(all_entropy) 
    low_uncert_idx = np.argmin(gmm.means_.flatten())
    prob = prob[:, low_uncert_idx]
    
    return prob, all_preds

def cal_pseudo_labels(clean_probs, cur_X, cur_y):
    clean_indices = (clean_probs >= args.gmm_threshold)
    pseudo_X = cur_X[clean_indices]
    pseudo_y = cur_y[clean_indices]
    pseudo_loader = data_processor.load_iter(pseudo_X, pseudo_y, args.batch_size, True, args.num_workers)
    return pseudo_loader

def model_adapt(model, test_X, adapt_loader, origin_loader, test_loader, eval_metrics, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    origin_iter = iter(origin_loader)
    criterion = torch.nn.CrossEntropyLoss()
    max_f1 = 0
    best_epoch = 0

    for epoch in range(100):
        if epoch % 5 == 0:
            clean_probs, pseudo_labels = cal_GMM_probs(model, test_loader, device)
            pseudo_loader = cal_pseudo_labels(clean_probs, test_X, pseudo_labels)
            pseudo_iter = iter(pseudo_loader)

        model.train()
        sum_origin_loss = 0
        sum_mmd_loss = 0
        sum_entropy_loss = 0
        sum_pseudo_loss = 0
        sum_count = 0

        for index, cur_data in enumerate(adapt_loader):
            try:
                cur_origin_data = next(origin_iter)
            except:
                origin_iter = iter(origin_loader)
                cur_origin_data = next(origin_iter)
            try:
                cur_pseudo_data = next(pseudo_iter)
            except:
                pseudo_iter = iter(pseudo_loader)
                cur_pseudo_data = next(pseudo_iter)
        
            cur_X, cur_y = cur_data[0].to(device), cur_data[1].to(device)
            origin_X, origin_y = cur_origin_data[0].to(device), cur_origin_data[1].to(device)
            pseudo_X, pseudo_y = cur_pseudo_data[0].to(device), cur_pseudo_data[1].to(device)
            optimizer.zero_grad()

            origin_outs, origin_features = model(origin_X)
            adapt_outs, adapt_features = model(cur_X)
            pseudo_outs, pseudo_features = model(pseudo_X)

            softmax_out = F.softmax(adapt_outs, dim=-1)
            msoftmax = softmax_out.mean(dim=0)

            classification_loss = criterion(origin_outs, origin_y)
            pseudo_loss = criterion(pseudo_outs, pseudo_y)
            min_entropy_loss = softmax_entropy(adapt_outs).mean(0) + torch.sum(msoftmax * torch.log(msoftmax + 1e-5))
            mmd_loss = cal_mmd_loss(origin_features, adapt_features)
            loss = pseudo_loss + min_entropy_loss + mmd_loss + classification_loss

            loss.backward()
            optimizer.step()
            sum_origin_loss += classification_loss.data.cpu().numpy() * origin_outs.shape[0]
            sum_mmd_loss += mmd_loss.data.cpu().numpy() * origin_outs.shape[0]
            sum_entropy_loss += min_entropy_loss.data.cpu().numpy() * origin_outs.shape[0]
            sum_pseudo_loss += pseudo_loss.data.cpu().numpy() * origin_outs.shape[0]
            sum_count += adapt_outs.shape[0]

        train_origin_loss = round(sum_origin_loss / sum_count, 3)
        train_mmd_loss    = round(sum_mmd_loss / sum_count, 3)
        train_entropy_loss = round(sum_entropy_loss / sum_count, 3)
        train_pseudo_loss = round(sum_pseudo_loss / sum_count)

        print(f"epoch {epoch} loss: origin={train_origin_loss}, entropy={train_entropy_loss}, mmd={train_mmd_loss}, pseudo={train_pseudo_loss}")
        epoch_result = model_eval(model, test_loader, eval_metrics, device)
        print(epoch_result)
        if epoch_result["F1-score"] > max_f1:
            max_f1 = epoch_result["F1-score"]
            best_epoch = epoch
    
        print(f"best epoch {best_epoch}: F1-score = {max_f1}")
        print("----------------------------")
    return max_f1, best_epoch

fix_seed = 2024
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

# Argument parser for command-line options, arguments, and sub-commands
parser = argparse.ArgumentParser(description="WFlib")
parser.add_argument("--dataset", type=str, required=True, default="CW", help="Dataset name")
parser.add_argument("--model", type=str, required=True, default="DF", help="Model name")
parser.add_argument("--device", type=str, default="cpu", help="Device, options=[cpu, cuda, cuda:x]")
parser.add_argument("--num_tabs", type=int, default=1, 
                    help="Maximum number of tabs opened by users while browsing")
parser.add_argument("--scenario", type=str, default="Closed-world", 
                    help="Attack scenario, options=[Closed-world, Open-world]")

# Input parameters
parser.add_argument("--train_file", type=str, default="train", help="train file")
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
parser.add_argument("--gmm_threshold", type=float, default=0.6, help="GMM threshold")
parser.add_argument("--model_save_name", type=str, default="proteus", help="Name used to save the model")

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
ckp_path = os.path.join(args.checkpoints, args.dataset, args.model)
os.makedirs(log_path, exist_ok=True)
out_file = os.path.join(log_path, f"{args.result_file}.json")

# Load training and validation data
print(f"loading test file: ", os.path.join(in_path, f"{args.test_file}.npz"))
train_X, train_y = data_processor.load_data(os.path.join(in_path, f"{args.train_file}.npz"), args.feature, args.seq_len, args.num_tabs)
test_X, test_y = data_processor.load_data(os.path.join(in_path, f"{args.test_file}.npz"), args.feature, args.seq_len, args.num_tabs)
num_classes = len(np.unique(test_y))

if args.num_tabs == 1:
    num_classes = len(np.unique(test_y))
    assert num_classes == test_y.max() + 1, "Labels are not continuous" # Ensure labels are continuous
else:
    num_classes = test_y.shape[1]

# Print dataset information
print(f"Train: X={train_X.shape}, y={train_y.shape}")
print(f"Test: X={test_X.shape}, y={test_y.shape}")
print(f"num_classes: {num_classes}")

# Load data into iterators
origin_iter = data_processor.load_iter(train_X, train_y, args.batch_size, True, args.num_workers)
adapt_iter = data_processor.load_iter(test_X, torch.zeros_like(test_y), args.batch_size, True, args.num_workers)
test_iter = data_processor.load_iter(test_X, test_y, args.batch_size, False, args.num_workers)

# Initialize model, optimizer, and loss function
if args.model in ["BAPM", "TMWF"]: # Assume num_tabs is known
    model = eval(f"models.{args.model}")(num_classes, args.num_tabs)
else:
    model = eval(f"models.{args.model}")(num_classes)

model.load_state_dict(torch.load(os.path.join(ckp_path, f"{args.load_name}.pth"), map_location="cpu"))
model.to(device)

# Evaluation
result = model_eval(model, test_iter, args.eval_metrics, device)
print("NoAdapt:")
print(result)

best_f1_score, best_f1_epoch = model_adapt(model, test_X, adapt_iter, origin_iter, test_iter, args.eval_metrics, device)

print("After Adapt:")
result = model_eval(model, test_iter, args.eval_metrics, device)
result["best_f1_score"] = best_f1_score
result["best_f1_epoch"] = best_f1_epoch
print(result)

model_save_file = os.path.join(ckp_path, f"{args.model_save_name}.pth")
torch.save(model.state_dict(), model_save_file)

with open(out_file, "w") as fp:
    json.dump(result, fp, indent=4)