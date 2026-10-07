"""
实验一（初级要求）：基于 kNN 的手写数字识别
数据集：Semeion 手写数字数据集（16x16 二值像素，共 1593 个样本，10 类数字 0~9）
要求：
  1. 编程实现 kNN 算法
  2. 采用留一法（Leave-One-Out, LOO）给出 k = 5, 9, 13 时的识别精度
"""

import numpy as np
from collections import Counter

DATA_PATH = r"..\data\semeion.data"


def load_semeion(path):
    """读取 semeion 数据集，返回 (X, y)。
    X: (n_samples, 256) 二值像素特征
    y: (n_samples,) 数字标签 0~9
    """
    data = np.loadtxt(path)
    X = data[:, :256]
    onehot = data[:, 256:]
    y = np.argmax(onehot, axis=1)
    return X, y


class KNN:
    """k 近邻分类器（留一法评估）。"""

    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        return self

    def predict(self, X_test):
        """对测试集做预测，使用欧氏距离。"""
        # 平方欧氏距离：||a-b||^2 = ||a||^2 + ||b||^2 - 2*a·b
        a2 = (X_test ** 2).sum(axis=1)[:, None]          # (n_test, 1)
        b2 = (self.X_train ** 2).sum(axis=1)[None, :]    # (1, n_train)
        cross = X_test @ self.X_train.T                  # (n_test, n_train)
        dist2 = a2 + b2 - 2.0 * cross
        dist2 = np.maximum(dist2, 0.0)                # 数值误差修正

        k = min(self.k, self.X_train.shape[0])
        idx = np.argsort(dist2, axis=1)[:, :k]        # 最近的 k 个训练样本
        knn_labels = self.y_train[idx]                # (n_test, k)

        preds = np.empty(X_test.shape[0], dtype=self.y_train.dtype)
        for i in range(X_test.shape[0]):
            counts = np.bincount(knn_labels[i], minlength=10)
            # 多数投票；平票时取类别编号最小者（确定性处理）
            preds[i] = np.argmax(counts)
        return preds

    def leave_one_out(self, X, y):
        """留一法评估：每次留一个样本作测试，其余作训练。"""
        n = X.shape[0]
        preds = np.empty(n, dtype=y.dtype)
        for i in range(n):
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            self.fit(X[mask], y[mask])
            preds[i] = self.predict(X[i:i + 1])[0]
        return preds


def accuracy(y_true, y_pred):
    return float(np.mean(y_true == y_pred))


def main():
    X, y = load_semeion(DATA_PATH)
    print("样本数:", X.shape[0], " 特征维度:", X.shape[1], " 类别数:", len(np.unique(y)))
    print("各类样本数:", dict(Counter(y)))
    print()

    ks = [5, 9, 13]
    print("=" * 46)
    print(f"{'k':>4} | {'正确数':>6} | {'错误数':>6} | {'识别精度(ACC)':>12}")
    print("-" * 46)
    for k in ks:
        preds = KNN(k).leave_one_out(X, y)
        acc = accuracy(y, preds)
        correct = int(np.sum(preds == y))
        wrong = int(np.sum(preds != y))
        print(f"{k:>4} | {correct:>6} | {wrong:>6} | {acc*100:>10.2f}%")


if __name__ == "__main__":
    main()
