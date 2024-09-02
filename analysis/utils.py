import torch
import numpy as np
from sklearn.mixture import GaussianMixture
from WFlib.tools import evaluator
from torch.utils.data import DataLoader
from WFlib.tools import data_processor
import torch.nn.functional as F

def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)

def build_loader(clean_probs, cur_X, cur_y, epoch, batch_size):
    GMM_threshold = min(0.85 + epoch*0.01, 0.95)
    clean_indices = (clean_probs >= GMM_threshold)
    noise_indices = (clean_probs < GMM_threshold)

    # 将伪标签数据合并为单个Tensor
    pseudo_X = cur_X[clean_indices]
    pseudo_y = cur_y[clean_indices]

    unique_values, counts = np.unique(pseudo_y, return_counts=True)
    print(f"Pseduo: count={pseudo_y.shape[0]}, class={len(unique_values)}")

    unlabeled_data = cur_X[noise_indices]
    unlabeled_labels = cur_y[noise_indices]

    pseudo_loader = data_processor.load_iter(pseudo_X, pseudo_y, batch_size, True, 8)
    unlabeled_loader = data_processor.load_iter(unlabeled_data, unlabeled_labels, batch_size, True, 8)

    return pseudo_loader, unlabeled_loader

def train_model2(pseudo_loader, unlabeled_loader, origin_loader, ares_model, ares_optimizer, df_model, df_optimizer, device):
    loss_ratio = 0.9
    df_model.train() 
    ares_model.train()

    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.0)
    unlabeled_iter = iter(unlabeled_loader)
    origin_iter = iter(origin_loader)
    
    # 使用伪标签进行训练DF
    for index, cur_data in enumerate(pseudo_loader):
        try:
            cur_unlabeled_data = next(unlabeled_iter)
        except:
            unlabeled_iter = iter(unlabeled_loader)
            cur_unlabeled_data = next(unlabeled_iter)
        
        try:
            cur_origin_data = next(origin_iter)
        except:
            origin_iter = iter(origin_loader)
            cur_origin_data = next(origin_iter)
        
        labeled_X, labeled_y = cur_data[0].to(device), cur_data[1].to(device)
        unlabeled_X = cur_unlabeled_data[0].to(device)
        origin_X, origin_y = cur_origin_data[0].to(device), cur_origin_data[1].to(device)
        
        df_optimizer.zero_grad()
        ares_optimizer.zero_grad()

        df_label_outs = df_model(labeled_X[...,:5000])
        df_unlabeled_outs = df_model(unlabeled_X[...,:5000])
        df_origin_outs = df_model(origin_X[...,:5000])

        ares_label_outs = ares_model(labeled_X)
        ares_unlabel_outs = ares_model(unlabeled_X)
        ares_origin_outs = ares_model(origin_X)
        
        df_loss = loss_ratio * criterion(df_label_outs, labeled_y) + (1-loss_ratio) * softmax_entropy(df_unlabeled_outs).mean(0) + (1-loss_ratio) * criterion(df_origin_outs, origin_y)
        ares_loss = loss_ratio * criterion(ares_label_outs, labeled_y) + (1-loss_ratio) * softmax_entropy(ares_unlabel_outs).mean(0) + (1-loss_ratio) * criterion(ares_origin_outs, origin_y)

        df_loss.backward()
        ares_loss.backward()

        df_optimizer.step()
        ares_optimizer.step()

def train_model(pseudo_loader, unlabeled_loader, origin_loader, cur_model, cur_optimizer, seq_len, device):
    loss_ratio = 0.9
    cur_model.train() # 设置cur_model.eval()避免BN和Dropout的更新
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.0)
    unlabeled_iter = iter(unlabeled_loader)
    origin_iter = iter(origin_loader)
    
    # 使用伪标签进行训练DF
    for index, cur_data in enumerate(pseudo_loader):
        try:
            cur_unlabeled_data = next(unlabeled_iter)
        except:
            unlabeled_iter = iter(unlabeled_loader)
            cur_unlabeled_data = next(unlabeled_iter)
        
        try:
            cur_origin_data = next(origin_iter)
        except:
            origin_iter = iter(origin_loader)
            cur_origin_data = next(origin_iter)
        
        cur_unlabeled_X = cur_unlabeled_data[0].to(device)
        cur_X, cur_y = cur_data[0].to(device), cur_data[1].to(device)
        origin_X, origin_y = cur_origin_data[0].to(device), cur_origin_data[1].to(device)
        
        cur_optimizer.zero_grad()

        outs = cur_model(cur_X[...,:seq_len])
        unlabeled_outs = cur_model(cur_unlabeled_X[...,:seq_len])
        origin_outs = cur_model(origin_X[...,:seq_len])
        
        labeled_loss = criterion(outs, cur_y)
        origin_loss = criterion(origin_outs, origin_y)
        unlabeled_loss = softmax_entropy(unlabeled_outs).mean(0)

        loss = loss_ratio * labeled_loss + (1-loss_ratio) * unlabeled_loss + 0.1 * origin_loss
        loss.backward()
        cur_optimizer.step()

def cal_GMM(train_iter, cur_model, seq_len, device):
    all_entropy = []
    all_preds = []
    with torch.no_grad():
        cur_model.eval()

        for index, cur_data in enumerate(train_iter):
            cur_X, cur_y = cur_data[0].to(device), cur_data[1].to(device)
            outs = cur_model(cur_X[...,:seq_len])
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
    prob = prob[:,gmm.means_.argmin()]
    return prob, all_preds

def eval_models(test_iter, df_model, ares_model, device):
    y_pred = []
    y_true = []
    fusion_ratio = 0.5
    with torch.no_grad():
        df_model.eval()
        ares_model.eval()

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
    return y_true, y_pred

def cal_GMM2(cur_iter, df_model, ares_model, device):
    # Fusion两个模型生成一个数据
    all_entropy = []
    all_preds = []
    fusion_ratio = 0.5
    with torch.no_grad():
        df_model.eval()
        ares_model.eval()

        for index, cur_data in enumerate(cur_iter):
            cur_X, cur_y = cur_data[0].to(device), cur_data[1].to(device)
            df_outs = df_model(cur_X[...,:5000])
            ares_outs = ares_model(cur_X)
            outs = fusion_ratio * df_outs + (1-fusion_ratio) * ares_outs
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
    prob = prob[:,gmm.means_.argmin()]
    return prob, all_preds

def random_sample(X, y, per_class_num=30):
    # 每个类别随机采样per_class_num个样本
    sampled_X = []
    sampled_y = []
    num_classes = len(np.unique(y))
    for web in range(num_classes):
        indices = torch.where(y == web)[0]
        if len(indices) > 0:
            sampled_indices = indices[torch.randperm(len(indices))[:per_class_num]]
        else:
            sampled_indices = indices
        sampled_X.append(X[sampled_indices])
        sampled_y.append(y[sampled_indices])
    sampled_X = torch.cat(sampled_X, dim=0)
    sampled_y = torch.cat(sampled_y, dim=0)
    return sampled_X, sampled_y
