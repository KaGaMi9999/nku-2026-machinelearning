"""实验二 初级 + 中级：手写线性回归（BGD / SGD）

- 读取 winequality-white.csv（分号分隔）
- 按 quality 分层 4:1 划分训练/测试集
- 用训练集统计量做 z-score 标准化，偏置项并入特征
- 手写批量梯度下降（BGD）与随机梯度下降（SGD）
- 不同学习率下的 MSE 收敛曲线对比，并选择较优学习率
- 用正规方程闭式解与 sklearn LinearRegression 校验手写实现

运行：python codes/linreg_bgd_sgd.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "reports" / "figs"
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

SEED = 42


class Tee:
    """同时输出到终端与文本文件。"""

    def __init__(self, path):
        self.file = open(path, "w", encoding="utf-8")

    def log(self, text=""):
        print(text)
        self.file.write(str(text) + "\n")
        self.file.flush()

    def __call__(self, text=""):
        self.log(text)

    def close(self):
        self.file.close()


def load_data():
    df = pd.read_csv(ROOT / "winequality-white.csv", sep=";")
    X = df.iloc[:, :11].to_numpy(dtype=float)
    y = df["quality"].to_numpy(dtype=float)
    return df, X, y


def split_and_scale(X, y, test_size=0.2, seed=SEED):
    """分层划分；标准化参数只由训练集确定。"""
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    mu = X_tr.mean(axis=0)
    sigma = X_tr.std(axis=0, ddof=0)
    return (X_tr - mu) / sigma, (X_te - mu) / sigma, y_tr, y_te, mu, sigma


def add_bias(X):
    return np.hstack([X, np.ones((X.shape[0], 1))])


def mse(Xb, y, theta):
    return float(np.mean((Xb @ theta - y) ** 2))


def bgd(Xb, y, lr, epochs, Xte_b=None, yte=None):
    """批量梯度下降，每个 epoch 记录训练/测试 MSE。"""
    m, n = Xb.shape
    theta = np.zeros(n)
    train_hist = np.empty(epochs)
    test_hist = np.empty(epochs)
    for epoch in range(epochs):
        grad = (2.0 / m) * (Xb.T @ (Xb @ theta - y))
        theta = theta - lr * grad
        tr = mse(Xb, y, theta)
        if not np.isfinite(tr) or tr > 1e6:  # 学习率过大导致发散
            train_hist[epoch:] = np.nan
            test_hist[epoch:] = np.nan
            break
        train_hist[epoch] = tr
        if Xte_b is not None:
            test_hist[epoch] = mse(Xte_b, yte, theta)
    return theta, train_hist, test_hist


def sgd(Xb, y, lr, epochs, seed=SEED, Xte_b=None, yte=None):
    """随机梯度下降：每个 epoch 打乱样本，逐样本更新一次。"""
    m, n = Xb.shape
    rng = np.random.default_rng(seed)
    theta = np.zeros(n)
    train_hist = np.empty(epochs)
    test_hist = np.empty(epochs)
    for epoch in range(epochs):
        for i in rng.permutation(m):
            xi = Xb[i]
            err = xi @ theta - y[i]
            theta = theta - lr * 2.0 * err * xi
        tr = mse(Xb, y, theta)
        if not np.isfinite(tr) or tr > 1e6:
            train_hist[epoch:] = np.nan
            test_hist[epoch:] = np.nan
            break
        train_hist[epoch] = tr
        if Xte_b is not None:
            test_hist[epoch] = mse(Xte_b, yte, theta)
    return theta, train_hist, test_hist


def plot_lr_curves(histories, xlabel, title, path, ylim=None, smooth=None):
    """histories: {lr: (train_hist, test_hist)}；smooth 为滑动平均窗口。"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for lr, (tr, te) in histories.items():
        label = f"lr={lr:g}"
        for ax, hist in zip(axes, (tr, te)):
            if smooth:
                ax.plot(np.arange(1, len(hist) + 1), hist, color="gray",
                        alpha=0.12, linewidth=1.0)
                smooth_vals = pd.Series(hist).rolling(smooth, center=True, min_periods=1).mean().to_numpy()
                ax.plot(np.arange(1, len(hist) + 1), smooth_vals, label=label, linewidth=1.6)
            else:
                ax.plot(np.arange(1, len(hist) + 1), hist, label=label, linewidth=1.4)
    for ax, name in zip(axes, ["训练集 MSE", "测试集 MSE"]):
        ax.set_yscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(name)
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=9)
        if ylim:
            ax.set_ylim(*ylim)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    log = Tee(ROOT / "reports" / "linreg_outputs.txt")
    df, X, y = load_data()

    log("=" * 70)
    log("实验二 初级：线性回归的 BGD / SGD 实现")
    log("=" * 70)
    log(f"数据规模: {df.shape[0]} 条样本, {df.shape[1] - 1} 个特征, 1 个目标 quality")
    log("quality 分布: " + ", ".join(f"{k}:{v}" for k, v in zip(*np.unique(y.astype(int), return_counts=True))))
    log(f"缺失值个数: {int(df.isna().sum().sum())}")
    log("各特征均值/标准差（训练集标准化前）:")
    log(f"{'特征':<22}{'均值':>10}{'标准差':>10}{'最小值':>10}{'最大值':>10}")
    for name, mean, std, lo, hi in zip(df.columns[:11], X.mean(0), X.std(0), X.min(0), X.max(0)):
        log(f"{name:<22}{mean:>10.4f}{std:>10.4f}{lo:>10.4f}{hi:>10.4f}")

    X_tr, X_te, y_tr, y_te, mu, sigma = split_and_scale(X, y)
    Xb_tr, Xb_te = add_bias(X_tr), add_bias(X_te)
    log(f"\n分层 4:1 划分: 训练 {len(y_tr)} 条, 测试 {len(y_te)} 条")
    log(f"训练集 quality 分布: {dict(zip(*[a.tolist() for a in np.unique(y_tr.astype(int), return_counts=True)]))}")
    log(f"测试集 quality 分布: {dict(zip(*[a.tolist() for a in np.unique(y_te.astype(int), return_counts=True)]))}")

    # ---------------- BGD ----------------
    bgd_lrs = [1e-3, 5e-3, 1e-2, 5e-2, 0.1, 0.2, 0.4]
    bgd_epochs = 2000
    log(f"\n{'-' * 70}\n[BGD] 全批量梯度下降, epochs={bgd_epochs}\n{'-' * 70}")
    log(f"{'学习率':>8}{'训练 MSE':>14}{'测试 MSE':>14}{'状态':>10}")
    bgd_hist, bgd_final = {}, {}
    for lr in bgd_lrs:
        theta, tr, te = bgd(Xb_tr, y_tr, lr, bgd_epochs, Xb_te, y_te)
        bgd_hist[lr] = (tr, te)
        bgd_final[lr] = (tr[-1], te[-1])
        state = "发散" if not np.isfinite(tr[-1]) else ("未收敛" if tr[-1] > 1.0 else "收敛")
        log(f"{lr:>8g}{tr[-1]:>14.6f}{te[-1]:>14.6f}{state:>10}")

    stable_bgd = {lr: v for lr, v in bgd_final.items() if np.isfinite(v[1])}
    best_bgd_lr = min(stable_bgd, key=lambda k: stable_bgd[k][1])
    log(f"BGD 较优学习率: {best_bgd_lr:g}（测试 MSE={bgd_final[best_bgd_lr][1]:.6f}）")

    plot_lr_curves(
        {lr: h for lr, h in bgd_hist.items() if np.isfinite(h[0][-1])},
        "epoch",
        "BGD 不同学习率下的 MSE 收敛曲线（lr=0.4 发散，已略去）",
        FIG_DIR / "bgd_lr_curves.png",
        ylim=(0.5, 50),
    )

    # ---------------- SGD ----------------
    sgd_lrs = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 2e-2]
    sgd_epochs = 40
    sgd_seeds = [42, 43, 44]
    log(f"\n{'-' * 70}\n[SGD] 随机梯度下降, epochs={sgd_epochs}, 曲线为 {len(sgd_seeds)} 个随机种子的平均\n{'-' * 70}")
    log(f"{'学习率':>8}{'训练 MSE':>16}{'测试 MSE':>18}{'状态':>8}")
    sgd_hist, sgd_final = {}, {}
    for lr in sgd_lrs:
        trs, tes = [], []
        for s in sgd_seeds:
            _, tr, te = sgd(Xb_tr, y_tr, lr, sgd_epochs, seed=s, Xte_b=Xb_te, yte=y_te)
            trs.append(tr)
            tes.append(te)
        tr_mean, te_mean = np.mean(trs, axis=0), np.mean(tes, axis=0)
        sgd_hist[lr] = (tr_mean, te_mean)
        sgd_final[lr] = (float(tr_mean[-1]), float(te_mean[-1]), float(np.std([t[-1] for t in tes])))
        state = "发散" if not np.isfinite(tr_mean[-1]) else ("未收敛" if tr_mean[-1] > 1.0 else "收敛")
        log(f"{lr:>8g}{sgd_final[lr][0]:>16.6f}{sgd_final[lr][1]:>13.6f}±{sgd_final[lr][2]:<4.4f}{state:>6}")

    stable_sgd = {lr: v for lr, v in sgd_final.items() if np.isfinite(v[1])}
    best_sgd_lr = min(stable_sgd, key=lambda k: stable_sgd[k][1])
    log(f"SGD 较优学习率: {best_sgd_lr:g}（测试 MSE={sgd_final[best_sgd_lr][1]:.6f}）")

    plot_lr_curves(
        {lr: h for lr, h in sgd_hist.items() if np.isfinite(h[0][-1])},
        "epoch",
        "SGD 不同学习率下的 MSE 收敛曲线（3 个种子平均；细灰线为原始值，粗线为 5 点滑动平均）",
        FIG_DIR / "sgd_lr_curves.png",
        ylim=(0.5, 3),
        smooth=5,
    )

    # ---------------- BGD vs SGD ----------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    btr, bte = bgd_hist[best_bgd_lr]
    str_, ste = sgd_hist[best_sgd_lr]
    kernel = lambda h: pd.Series(h).rolling(5, center=True, min_periods=1).mean().to_numpy()
    axes[0].plot(np.arange(1, len(btr) + 1), btr, label=f"BGD (lr={best_bgd_lr:g})")
    axes[0].plot(np.arange(1, len(str_) + 1), kernel(str_),
                 label=f"SGD (lr={best_sgd_lr:g}, 5 点平滑)")
    axes[1].plot(np.arange(1, len(bte) + 1), bte, label=f"BGD (lr={best_bgd_lr:g})")
    axes[1].plot(np.arange(1, len(ste) + 1), kernel(ste),
                 label=f"SGD (lr={best_sgd_lr:g}, 5 点平滑)")
    for ax, name in zip(axes, ["训练集 MSE", "测试集 MSE"]):
        ax.set_yscale("log")
        ax.set_xlabel("epoch")
        ax.set_ylabel(name)
        ax.grid(alpha=0.3, which="both")
        ax.set_xlim(0, 50)
        ax.set_ylim(0.5, 3)
        ax.legend()
    fig.suptitle("BGD 与 SGD 收敛过程对比（各自较优学习率，前 50 个 epoch）")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "bgd_vs_sgd.png", dpi=150)
    plt.close(fig)

    # ---------------- 闭式解与 sklearn 校验 ----------------
    theta_ols = np.linalg.lstsq(Xb_tr, y_tr, rcond=None)[0]
    ols_tr, ols_te = mse(Xb_tr, y_tr, theta_ols), mse(Xb_te, y_te, theta_ols)

    sk = LinearRegression().fit(X_tr, y_tr)
    sk_tr, sk_te = mse(Xb_tr, y_tr, np.append(sk.coef_, sk.intercept_)), mse(Xb_te, y_te, np.append(sk.coef_, sk.intercept_))

    log(f"\n{'-' * 70}\n[对比] 闭式解（正规方程）与 sklearn LinearRegression\n{'-' * 70}")
    log(f"{'模型':<26}{'训练 MSE':>14}{'测试 MSE':>14}")
    log(f"{'正规方程闭式解':<24}{ols_tr:>14.6f}{ols_te:>14.6f}")
    log(f"{'sklearn LinearRegression':<22}{sk_tr:>14.6f}{sk_te:>14.6f}")
    log(f"{'手写 BGD(最优 lr)':<23}{bgd_final[best_bgd_lr][0]:>14.6f}{bgd_final[best_bgd_lr][1]:>14.6f}")
    log(f"{'手写 SGD(最优 lr)':<23}{sgd_final[best_sgd_lr][0]:>14.6f}{sgd_final[best_sgd_lr][1]:>14.6f}")
    log(f"手写闭式解与 sklearn 系数最大绝对差: {np.max(np.abs(theta_ols - np.append(sk.coef_, sk.intercept_))):.3e}")

    # 用与 BGD 相同的设置重新取一次 theta 做系数对比
    theta_bgd, _, _ = bgd(Xb_tr, y_tr, best_bgd_lr, bgd_epochs, Xb_te, y_te)
    log(f"手写 BGD 与正规方程系数的最大绝对差: {np.max(np.abs(theta_bgd - theta_ols)):.3e}")

    # 训练/测试 MSE 与目标方差对比
    log(f"训练集 quality 方差: {np.var(y_tr):.6f}, 测试集 quality 方差: {np.var(y_te):.6f}")
    log(f"正规方程模型 R^2（测试集）: {1 - ols_te / np.var(y_te):.4f}")

    log("\n图已保存: reports/figs/bgd_lr_curves.png, sgd_lr_curves.png, bgd_vs_sgd.png")
    log.close()


if __name__ == "__main__":
    main()
