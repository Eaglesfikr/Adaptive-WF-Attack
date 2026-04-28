import torch
from torch import nn
import torch.nn.functional as F
import os
import sys
import random
import argparse
import numpy as np
from WFlib import models
from WFlib.tools import data_processor,evaluator

# =========================
# 1️⃣ Conv Block（原始 DF）
# =========================
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 pool_size, pool_stride, dropout_p, activation):
        super().__init__()
        padding = kernel_size // 2

        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, stride,
                      padding=padding, bias=False),
            nn.BatchNorm1d(out_channels),
            activation(inplace=True),

            nn.Conv1d(out_channels, out_channels, kernel_size, stride,
                      padding=padding, bias=False),
            nn.BatchNorm1d(out_channels),
            activation(inplace=True),

            nn.MaxPool1d(pool_size, pool_stride),
            nn.Dropout(p=dropout_p)
        )

    def forward(self, x):
        return self.block(x)


# =========================
# 2️⃣ DF Backbone（时域）
# =========================
class DF(nn.Module):
    def __init__(self):
        super().__init__()

        filter_num = [32, 64, 128, 256]
        kernel_size = 8
        conv_stride = 1
        pool_stride = 4
        pool_size = 8
        length_after = 18  # ⚠️ 依赖输入长度

        self.feature_extraction = nn.Sequential(
            ConvBlock(1, filter_num[0], kernel_size, conv_stride, pool_size, pool_stride, 0.1, nn.ELU),
            ConvBlock(filter_num[0], filter_num[1], kernel_size, conv_stride, pool_size, pool_stride, 0.1, nn.ReLU),
            ConvBlock(filter_num[1], filter_num[2], kernel_size, conv_stride, pool_size, pool_stride, 0.1, nn.ReLU),
            ConvBlock(filter_num[2], filter_num[3], kernel_size, conv_stride, pool_size, pool_stride, 0.1, nn.ReLU),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(filter_num[3] * length_after, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.7),

            nn.Linear(512, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )

    def forward(self, x):
        x = self.feature_extraction(x)
        feat = self.classifier(x)   # (B, 512)
        return feat


# =========================
# 3️⃣ SpectralConv1d（频域）
# =========================
class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, modes1, out_dim=512):
        super().__init__()
        self.modes1 = modes1

        self.proj = nn.Sequential(
            nn.Linear(modes1 * 2, 256),
            nn.ReLU(),
            nn.Linear(256, out_dim)
        )

    def forward(self, x):
        # x: (B, C, L)
        x_ft = torch.fft.rfft(x, norm='ortho')

        r = x_ft[:, :, :self.modes1].abs()
        p = x_ft[:, :, :self.modes1].angle()

        ef = torch.cat([r, p], dim=-1)  # (B, C, 2*modes1)
        ef = ef.mean(dim=1)             # (B, 2*modes1)

        ef = self.proj(ef)              # (B, 512)
        return ef


# =========================
# 4️⃣ TF Encoder（融合核心）
# =========================
class TFEncoder(nn.Module):
    def __init__(self, modes1=16):
        super().__init__()

        self.time_encoder = DF()
        self.freq_encoder = SpectralConv1d(1, modes1, 512)

        self.fusion = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.LayerNorm(512)
        )

    def forward(self, x):
        feat_t = self.time_encoder(x)   # (B, 512)
        feat_f = self.freq_encoder(x)   # (B, 512)

        f = torch.cat([feat_t, feat_f], dim=-1)  # (B, 1024)
        f = self.fusion(f)  # (B, 512)

        return f

# =========================
# 5️⃣ 完整模型（带分类头）
# =========================
class TFModel(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.encoder = TFEncoder()
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        feat = self.encoder(x)
        out = self.classifier(feat)
        return out, feat

# def load_data(data_path, feature_type, seq_len, num_tab=1):
#     data = np.load(data_path)
#     X = data["X"]
#     y = data["y"]

#     if feature_type == "DIR":
#         X = np.sign(X)  # Directional feature
#         X = length_align(X, seq_len)
#         X = torch.tensor(X[:,np.newaxis], dtype=torch.float32)
#     elif feature_type == "DT":
#         X = length_align(X, seq_len)
#         X = torch.tensor(X[:,np.newaxis], dtype=torch.float32)
#     elif feature_type == "DT2":
#         X_dir = np.sign(X)
#         X_time = np.abs(X)
#         X_time = np.diff(X_time)
#         X_time[X_time < 0] = 0  # Ensure no negative values
#         X_dir = length_align(X_dir, seq_len)[:, np.newaxis]
#         X_time = length_align(X_time, seq_len)[:, np.newaxis]
#         X = np.concatenate([X_dir, X_time], axis=1)
#         X = torch.tensor(X, dtype=torch.float32)
#     elif feature_type == "TAM":
#         X = length_align(X, seq_len)
#         X = torch.tensor(X[:,np.newaxis], dtype=torch.float32)
#     elif feature_type in ["TAF", "MTAF"]:
#         X = length_align(X, seq_len)
#         X = torch.tensor(X, dtype=torch.float32)
#     elif feature_type == "Origin":
#         X = length_align(X, seq_len)
#         return X, y
#     else:
#         raise ValueError(f"Feature type {feature_type} is not matched.")
    
#     if num_tab == 1:
#         y = torch.tensor(y, dtype=torch.int64)
#     else:
#         y = torch.tensor(y, dtype=torch.float32)

#     return X, y

# def load_iter(X, y, batch_size, is_train=True, num_workers=8, weight_sample=False):
#     if weight_sample:
#         class_sample_count = np.unique(y.numpy(), return_counts=True)[1]
#         weight = 1.0 / class_sample_count
#         samples_weight = weight[y.numpy()]
#         samples_weight = torch.from_numpy(samples_weight)
#         sampler = torch.utils.data.sampler.WeightedRandomSampler(
#             samples_weight, len(samples_weight)
#         )
#         dataset = torch.utils.data.TensorDataset(X, y)
#         return torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers)
#     dataset = torch.utils.data.TensorDataset(X, y)
#     return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=is_train, drop_last=is_train, num_workers=num_workers)


def model_train(
    model,
    optimizer,
    train_iter,
    valid_iter,
    loss_name,
    save_metric,
    eval_metrics,
    train_epochs,
    out_file,
    num_classes,
    device,
    lradj
):

    criterion = torch.nn.CrossEntropyLoss()

    scheduler = None
    if lradj != "None":
        scheduler = eval(f"torch.optim.lr_scheduler.{lradj}")(
            optimizer, step_size=30, gamma=0.74
        )

    metric_best_value = 0
    best_epoch = 0

    for epoch in range(train_epochs):
        model.train()

        sum_loss = 0
        sum_count = 0

        for cur_X, cur_y in train_iter:
            cur_X, cur_y = cur_X.to(device), cur_y.to(device)

            optimizer.zero_grad()

            # ======================
            # TFModel forward
            # ======================
            outs, feat = model(cur_X)

            loss = criterion(outs, cur_y)

            loss.backward()
            optimizer.step()

            sum_loss += loss.item() * outs.size(0)
            sum_count += outs.size(0)

        train_loss = sum_loss / sum_count
        print(f"epoch {epoch}: train_loss = {train_loss:.4f}")

        # ======================
        # validation
        # ======================
        model.eval()
        valid_pred = []
        valid_true = []

        with torch.no_grad():
            for cur_X, cur_y in valid_iter:
                cur_X = cur_X.to(device)

                outs, _ = model(cur_X)

                pred = torch.argmax(outs, dim=1)

                valid_pred.append(pred.cpu().numpy())
                valid_true.append(cur_y.numpy())

        valid_pred = np.concatenate(valid_pred)
        valid_true = np.concatenate(valid_true)

        valid_result = evaluator.measurement(valid_true, valid_pred, eval_metrics)

        print(f"{epoch}: {valid_result}")

        if valid_result[save_metric] > metric_best_value:
            metric_best_value = valid_result[save_metric]
            best_epoch = epoch
            torch.save(model.state_dict(), out_file)

        print(f"best epoch {best_epoch}: {save_metric}={metric_best_value}")

        if scheduler is not None:
            scheduler.step()

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
        train_file = "train"
        valid_file = "valid"
        num_workers = 10
        loss = "CrossEntropyLoss"
        lradj = "None"
        checkpoints = "./checkpoints3/"
        load_file = None

    args = Args()
    device = torch.device(args.device)

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


    # Load training and validation data(为了满足不同格式的数据集)
    print(f"loading train file: ", os.path.join(in_path, f"{args.train_file}.npz"))
    train_X, train_y = data_processor.load_data(os.path.join(in_path, f"{args.train_file}.npz"), args.feature, args.seq_len, args.num_tabs)
    valid_X, valid_y = data_processor.load_data(os.path.join(in_path, f"{args.valid_file}.npz"), args.feature, args.seq_len, args.num_tabs)
    
    if args.num_tabs == 1:
        num_classes = len(np.unique(train_y))
        assert num_classes == train_y.max() + 1, "Labels are not continuous" # Ensure labels are continuous
    else:
        num_classes = train_y.shape[1]
    
    # Print dataset information
    print(f"Train: X={train_X.shape}, y={train_y.shape}")
    print(f"Valid: X={valid_X.shape}, y={valid_y.shape}")
    print(f"num_classes: {num_classes}")
    
    # Load data into iterators同时解决类别不平衡的问题
    train_iter = data_processor.load_iter(train_X, train_y, args.batch_size, True, args.num_workers)
    valid_iter = data_processor.load_iter(valid_X, valid_y, args.batch_size, False, args.num_workers)
    
    model = TFModel(num_classes=102).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    
    model_train(
        model=model,
        optimizer=optimizer,
        train_iter=train_iter,
        valid_iter=valid_iter,
        loss_name=args.loss,
        save_metric=args.save_metric,
        eval_metrics=args.eval_metrics,
        train_epochs=args.train_epochs,
        out_file=out_file,
        num_classes=10,
        device=device,
        lradj=args.lradj
    )