import os
import sys
import torch
import random
import argparse
import numpy as np
from WFlib import models
from WFlib.tools import data_processor, model_utils

# Set a fixed seed for reproducibility
fix_seed = 1013
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)


# ===== 修复 FFT 数据维度 =====
def fix_shape(x):
    if len(x.shape) == 4 and x.shape[2] == 1:
        x = x.squeeze(2)  # [B,1,1,L] -> [B,1,L]
    return x


# my adding
if __name__ == "__main__":
    class Args:
        dataset = "TemporalDrift"
        model = "DF"
        device = "cuda:0"
        feature = "DT"
        seq_len = 5000
        train_epochs = 30
        batch_size = 128
        learning_rate = 2e-3
        optimizer = "Adamax"
        eval_metrics = ["Accuracy", "Precision", "Recall", "F1-score"]
        save_metric = "F1-score"
        save_name = "max_f1_DT"
    
        # 默认参数
        num_tabs = 1
        train_file = "train_fft"
        valid_file = "valid_fft"
        num_workers = 10
        loss = "CrossEntropyLoss"
        lradj = "None"
        checkpoints = "./checkpoints2/"
        load_file = None

    args = Args()

    #my adding above
    
    # Ensure the specified device is available
    if args.device.startswith("cuda"):
        assert torch.cuda.is_available(), f"The specified device {args.device} does not exist"
    device = torch.device(args.device)
    
    # Define paths for dataset and checkpoints
    in_path = os.path.join("./datasets", args.dataset)
    if not os.path.exists(in_path):
        raise FileNotFoundError(f"The dataset path does not exist: {in_path}")
    ckp_path = os.path.join(args.checkpoints, args.dataset, args.model)
    os.makedirs(ckp_path, exist_ok=True)
    
    out_file = os.path.join(ckp_path, f"{args.save_name}.pth")
    if os.path.exists(out_file):
        print(f"{out_file} has been generated.")
        sys.exit(1)
    
    # Load training and validation data
    print(f"loading train file: ", os.path.join(in_path, f"{args.train_file}.npz"))
    train_X, train_y = data_processor.load_data(os.path.join(in_path, f"{args.train_file}.npz"), args.feature, args.seq_len, args.num_tabs)
    valid_X, valid_y = data_processor.load_data(os.path.join(in_path, f"{args.valid_file}.npz"), args.feature, args.seq_len, args.num_tabs)
    train_X = fix_shape(train_X)
    valid_X = fix_shape(valid_X)
    print("After fix:")
    print("Train:", train_X.shape)
    print("Valid:", valid_X.shape)
    
    if args.num_tabs == 1:
        num_classes = len(np.unique(train_y))
        assert num_classes == train_y.max() + 1, "Labels are not continuous" # Ensure labels are continuous
    else:
        num_classes = train_y.shape[1]
    
    # Print dataset information
    print(f"Train: X={train_X.shape}, y={train_y.shape}")
    print(f"Valid: X={valid_X.shape}, y={valid_y.shape}")
    print(f"num_classes: {num_classes}")
    
    # Load data into iterators
    train_iter = data_processor.load_iter(train_X, train_y, args.batch_size, True, args.num_workers)
    valid_iter = data_processor.load_iter(valid_X, valid_y, args.batch_size, False, args.num_workers)
    
    # Initialize model, optimizer, and loss function
    if args.model in ["BAPM", "TMWF"]: # Assume num_tabs is known
        model = eval(f"models.{args.model}")(num_classes, args.num_tabs)
    else:
        model = eval(f"models.{args.model}")(num_classes)
    optimizer = eval(f"torch.optim.{args.optimizer}")(model.parameters(), lr=args.learning_rate)
    
    if args.load_file is None:
        print("No pre-trained model")
    else:
        print("Loading the pretrained model in ", args.load_file)
        checkpoint = torch.load(args.load_file)
    
        # for k in list(checkpoint.keys()):
        #     if k.startswith('backbone.'):
        #         if k.startswith('backbone') and not k.startswith('backbone.fc'):
        #             checkpoint[k[len("backbone."):]] = checkpoint[k]
        #     del checkpoint[k]
    
        model.load_state_dict(checkpoint, strict=False)
        # assert log.missing_keys == ['fc.weight', 'fc.bias']
    
    model.to(device)
    
    # Train the model
    model_utils.model_train(
        model, 
        optimizer, 
        train_iter, 
        valid_iter, 
        args.loss, 
        args.save_metric, 
        args.eval_metrics, 
        args.train_epochs,
        out_file,
        num_classes,
        args.num_tabs,
        device,
        args.lradj
    )