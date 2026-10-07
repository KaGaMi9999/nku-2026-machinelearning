# -*- coding: utf-8 -*-
"""
实验一（高级要求）：旋转数据增强 + CNN 手写体识别
数据集：Semeion 手写数字（16×16 二值，1593 样本，10 类数字 0~9）
增强：对训练集做左上（逆时针，正角）与左下（顺时针，负角）两个方向的旋转扩充
模型：PyTorch CNN（Conv+ReLU+MaxPool ×2 → FC → 10 类）
"""

import numpy as np
from scipy.ndimage import rotate
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

DATA_PATH = r"..\data\semeion.data"
ROT_ANGLES = [+10, +15, -10, -15]   # 左上(正角=逆时针)、左下(负角=顺时针)各两个角度


def load_semeion(path):
    data = np.loadtxt(path)
    X = data[:, :256].astype(np.float32)
    y = np.argmax(data[:, 256:], axis=1).astype(np.int64)
    return X, y


def stratified_split(y, test_ratio=0.2, seed=42):
    """按类别分层划分，保证每类训练/测试比例一致。"""
    rng = np.random.RandomState(seed)
    tr, te = [], []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        nte = max(1, int(round(len(idx) * test_ratio)))
        te.extend(idx[:nte])
        tr.extend(idx[nte:])
    return np.array(tr), np.array(te)


def rotate_augment(images, angles):
    """对每张图按给定角度集合做旋转，返回 (N*len(angles), 16, 16)。"""
    out = []
    for img in images:
        for a in angles:
            out.append(rotate(img, a, reshape=False, order=1,
                              mode="constant", cval=0.0))
    return np.asarray(out, dtype=np.float32)


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(64 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 10)
        self.drop = nn.Dropout(0.3)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))   # 16 -> 8
        x = self.pool(F.relu(self.conv2(x)))   # 8 -> 4
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        return self.fc2(x)


def to_dataset(X, y):
    Xt = torch.tensor(X).view(-1, 1, 16, 16)
    yt = torch.tensor(y, dtype=torch.long)
    return TensorDataset(Xt, yt)


def train(model, loader, epochs, lr=1e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss()
    model.train()
    for ep in range(epochs):
        total, corr, loss_sum = 0, 0, 0.0
        for xb, yb in loader:
            opt.zero_grad()
            out = model(xb)
            loss = lossf(out, yb)
            loss.backward()
            opt.step()
            total += yb.size(0)
            corr += (out.argmax(1) == yb).sum().item()
            loss_sum += loss.item() * yb.size(0)
        if ep == 0 or (ep + 1) % 5 == 0:
            print(f"  epoch {ep+1:2d}/{epochs}  loss={loss_sum/total:.4f}  train_acc={corr/total*100:.2f}%")
    return model


def evaluate(model, X, y):
    model.eval()
    Xt = torch.tensor(X).view(-1, 1, 16, 16)
    with torch.no_grad():
        pred = model(Xt).argmax(1).numpy()
    return float((pred == y).mean())


def run_experiment(name, Xtr, ytr, Xte, yte, epochs=40, seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = CNN()
    loader = DataLoader(to_dataset(Xtr, ytr), batch_size=64, shuffle=True)
    print(f"[{name}] 训练样本数: {len(ytr)}")
    train(model, loader, epochs)
    acc = evaluate(model, Xte, yte)
    print(f"[{name}] 测试集 ACC = {acc*100:.2f}%\n")
    return model, acc


def main():
    X, y = load_semeion(DATA_PATH)
    X = X.reshape(-1, 16, 16)
    print(f"样本数: {len(y)}  图像: 16×16  类别: 10")
    tr, te = stratified_split(y, test_ratio=0.2, seed=42)
    Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]
    print(f"训练集: {len(tr)}  测试集: {len(te)}\n")

    # 训练两个模型
    base_model, acc_base = run_experiment("基线（无增强）", Xtr, ytr, Xte, yte)

    Xtr_aug = np.concatenate([Xtr, rotate_augment(Xtr, ROT_ANGLES)], axis=0)
    ytr_aug = np.concatenate([ytr] * (len(ROT_ANGLES) + 1))
    aug_model, acc_aug = run_experiment("旋转增强", Xtr_aug, ytr_aug, Xte, yte)

    # 旋转鲁棒性：用旋转后的测试集评估两个模型
    Xte_rot = rotate_augment(Xte, ROT_ANGLES)
    yte_rot = np.concatenate([yte] * len(ROT_ANGLES))
    acc_base_rot = evaluate(base_model, Xte_rot, yte_rot)
    acc_aug_rot = evaluate(aug_model, Xte_rot, yte_rot)

    print("=" * 56)
    print(f"{'模型':<12} | {'原始测试集':>10} | {'旋转测试集':>10}")
    print("-" * 56)
    print(f"{'基线(无增强)':<12} | {acc_base*100:>9.2f}% | {acc_base_rot*100:>9.2f}%")
    print(f"{'旋转增强':<12} | {acc_aug*100:>9.2f}% | {acc_aug_rot*100:>9.2f}%")
    print("-" * 56)
    print(f"旋转鲁棒性提升 = {(acc_aug_rot - acc_base_rot)*100:+.2f} 个百分点")


if __name__ == "__main__":
    main()
