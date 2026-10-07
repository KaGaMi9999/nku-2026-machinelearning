# -*- coding: utf-8 -*-
"""
实验一（中级要求）：基于 KNN 的人脸识别 + 与机器学习包（sklearn）对比
数据集：ORL (AT&T) 人脸库，40 人 × 10 张 = 400 张，92×112 灰度（10304 维）
下载来源：https://www.cl.cam.ac.uk/research/dtg/attarchive/pub/data/att_faces.zip
指标：ACC（精度）、NMI（归一化互信息）、CEN（混淆熵）
方法：手写 KNN 与 sklearn KNeighborsClassifier 在同一训练/测试划分下对比
"""

import os
import sys

# 若 sklearn 未全局安装，则从脚本旁的 _deps 目录加载（用 --target 安装的第三方依赖）
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "_deps"))

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score,
                             normalized_mutual_info_score,
                             confusion_matrix)

DATA_DIR = r"..\data\orl"
CACHE_X = r"..\data\orl_faces.npy"
CACHE_Y = r"..\data\orl_labels.npy"


class KNN:
    """手写 k 近邻分类器（欧氏距离 + 多数投票，平票取类别编号最小者）。"""

    def __init__(self, k=1):
        self.k = k

    def fit(self, X, y):
        self.X_train = np.asarray(X, dtype=float)
        self.y_train = np.asarray(y)
        self.n_classes = int(np.max(self.y_train)) + 1
        return self

    def predict(self, X_test):
        X_test = np.asarray(X_test, dtype=float)
        # 平方欧氏距离：||a-b||^2 = ||a||^2 + ||b||^2 - 2*a·b
        a2 = (X_test ** 2).sum(axis=1)[:, None]
        b2 = (self.X_train ** 2).sum(axis=1)[None, :]
        cross = X_test @ self.X_train.T
        dist2 = np.maximum(a2 + b2 - 2.0 * cross, 0.0)

        k = min(self.k, self.X_train.shape[0])
        idx = np.argsort(dist2, axis=1)[:, :k]
        knn_labels = self.y_train[idx]

        preds = np.empty(X_test.shape[0], dtype=self.y_train.dtype)
        for i in range(X_test.shape[0]):
            counts = np.bincount(knn_labels[i], minlength=self.n_classes)
            preds[i] = np.argmax(counts)
        return preds


def confusion_entropy(y_true, y_pred):
    """混淆熵 CEN（Wei et al. 2010），范围 [0,1]，越小表示混淆越少、分类越好。

    基于混淆矩阵 C（K×K），C_ij = 真实类 i 被预测为类 j 的数量，S = 总样本数。
    对每个类 j：
      P_{j,k}^{j} = C_jk / (第 j 行和 + 第 j 列和)   （真实 j 判为 k，关于类 j）
      P_{k,j}^{j} = C_kj / (第 j 行和 + 第 j 列和)   （真实 k 判为 j，关于类 j）
      CEN_j = - Σ_{k≠j} [ P_{j,k}^{j}·log_{2(K-1)} P_{j,k}^{j}
                          + P_{k,j}^{j}·log_{2(K-1)} P_{k,j}^{j} ]
      CEN   = Σ_j P_j · CEN_j，其中 P_j = (第 j 行和 + 第 j 列和) / (2S)
    """
    cm = confusion_matrix(y_true, y_pred).astype(float)
    K = cm.shape[0]
    if K < 2:
        return 0.0
    S = cm.sum()
    log_base = np.log(2 * (K - 1))
    cen = 0.0
    for j in range(K):
        denom_j = cm[j, :].sum() + cm[:, j].sum()
        if denom_j <= 0:
            continue
        weight = denom_j / (2.0 * S)
        cen_j = 0.0
        for k in range(K):
            if k == j:
                continue
            p1 = cm[j, k] / denom_j       # P_{j,k}^{j}
            if p1 > 0:
                cen_j += -p1 * np.log(p1) / log_base
            p2 = cm[k, j] / denom_j       # P_{k,j}^{j}
            if p2 > 0:
                cen_j += -p2 * np.log(p2) / log_base
        cen += weight * cen_j
    return float(cen)


def read_pgm(path):
    """读取 PGM P5 二进制灰度图，返回 (height, width) 的 uint8 数组。"""
    with open(path, "rb") as f:
        magic = f.readline().strip()
        assert magic == b"P5", f"不支持的 PGM 格式: {magic}"
        line = f.readline().strip()
        while line.startswith(b"#"):          # 跳过注释行
            line = f.readline().strip()
        width, height = map(int, line.split())
        maxval = int(f.readline().strip())
        assert maxval == 255, f"非 8-bit 灰度: {maxval}"
        pixels = f.read(width * height)
    return np.frombuffer(pixels, dtype=np.uint8).reshape(height, width)


def load_orl(data_dir=DATA_DIR):
    """加载 ORL 人脸数据集，返回 (X, y)；首次解析后缓存为 .npy。"""
    if os.path.exists(CACHE_X) and os.path.exists(CACHE_Y):
        return np.load(CACHE_X), np.load(CACHE_Y)
    X, y = [], []
    for subj in range(1, 41):
        subdir = os.path.join(data_dir, f"s{subj}")
        for img in range(1, 11):
            arr = read_pgm(os.path.join(subdir, f"{img}.pgm"))
            X.append(arr.ravel())
            y.append(subj - 1)
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    np.save(CACHE_X, X)
    np.save(CACHE_Y, y)
    return X, y


def evaluate(y_true, y_pred):
    return {
        "ACC": accuracy_score(y_true, y_pred),
        "NMI": normalized_mutual_info_score(y_true, y_pred),
        "CEN": confusion_entropy(y_true, y_pred),
    }


def main():
    print("正在加载 ORL 人脸数据集 ...")
    X, y = load_orl()
    print(f"样本数: {X.shape[0]}  特征维度: {X.shape[1]}  类别数(人): {len(np.unique(y))}")
    print()

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42)
    print(f"训练集: {X_tr.shape[0]}  测试集: {X_te.shape[0]}")
    print()

    ks = [1, 3, 5]
    sep = "=" * 60
    print(sep)
    print(f"{'k':>3} | {'模型':>14} | {'ACC':>8} | {'NMI':>8} | {'CEN':>8}")
    print("-" * 60)
    for k in ks:
        pred_mine = KNN(k).fit(X_tr, y_tr).predict(X_te)
        m = evaluate(y_te, pred_mine)

        sk = KNeighborsClassifier(n_neighbors=k, metric="euclidean")
        sk.fit(X_tr, y_tr)
        s = evaluate(y_te, sk.predict(X_te))

        print(f"{k:>3} | {'手写KNN':>14} | {m['ACC']*100:>7.2f}% | {m['NMI']:>8.4f} | {m['CEN']:>8.4f}")
        print(f"{k:>3} | {'sklearn':>14} | {s['ACC']*100:>7.2f}% | {s['NMI']:>8.4f} | {s['CEN']:>8.4f}")
        print("-" * 60)

    # 指标正确性自检
    print("\n[自检] CEN 边界验证：")
    yt = np.arange(0, 5).repeat(2)
    print(f"  完美预测 CEN = {confusion_entropy(yt, yt):.4f}（应为 0）")
    np.random.seed(0)
    yr = np.random.randint(0, 5, size=len(yt))
    print(f"  随机预测 CEN = {confusion_entropy(yt, yr):.4f}（应接近 1）")


if __name__ == "__main__":
    main()
