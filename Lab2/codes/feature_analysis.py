"""实验二 中级：特征影响分析与相关特征（共线性）实验

- 标准化线性回归模型的系数大小与符号
- 置换重要性：打乱单个特征后测试 MSE 的增量
- 训练集 Pearson 相关系数热力图，选出最相关的一对特征
- 对比“两者都保留 / 去掉其一”时的系数与测试 MSE

运行：python codes/feature_analysis.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "reports" / "figs"
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

SEED = 42


class Tee:
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


def load_split():
    df = pd.read_csv(ROOT / "winequality-white.csv", sep=";")
    names = list(df.columns[:11])
    X = df.iloc[:, :11].to_numpy(dtype=float)
    y = df["quality"].to_numpy(dtype=float)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    mu, sigma = X_tr.mean(axis=0), X_tr.std(axis=0, ddof=0)
    return names, (X_tr - mu) / sigma, (X_te - mu) / sigma, y_tr, y_te


def fit_ols(X, y):
    Xb = np.hstack([X, np.ones((X.shape[0], 1))])
    return np.linalg.lstsq(Xb, y, rcond=None)[0]


def mse(X, y, theta):
    Xb = np.hstack([X, np.ones((X.shape[0], 1))])
    return float(np.mean((Xb @ theta - y) ** 2))


def main():
    log = Tee(ROOT / "reports" / "feature_outputs.txt")
    names, X_tr, X_te, y_tr, y_te = load_split()

    log("=" * 70)
    log("实验二 中级：特征影响与共线性分析")
    log("=" * 70)
    theta = fit_ols(X_tr, y_tr)
    coef = theta[:-1]
    log(f"训练集 MSE={mse(X_tr, y_tr, theta):.6f}, 测试集 MSE={mse(X_te, y_te, theta):.6f}")
    log("\n标准化回归系数（按绝对值降序）:")
    log(f"{'特征':<24}{'系数':>12}{'|系数|':>12}")
    for idx in np.argsort(-np.abs(coef)):
        log(f"{names[idx]:<22}{coef[idx]:>12.4f}{abs(coef[idx]):>12.4f}")
    log(f"偏置项 {theta[-1]:.4f}")

    # ---- 系数图 ----
    order = np.argsort(coef)
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#d55e5e" if c < 0 else "#4c72b0" for c in coef[order]]
    ax.barh(np.array(names)[order], coef[order], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("标准化回归系数")
    ax.set_title("线性回归标准化系数（按大小排序）")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "feature_coefficients.png", dpi=150)
    plt.close(fig)

    # ---- 置换重要性 ----
    rng = np.random.default_rng(SEED)
    base = mse(X_te, y_te, theta)
    repeats = 30
    imp_mean = np.zeros(len(names))
    imp_std = np.zeros(len(names))
    for j in range(len(names)):
        deltas = []
        for _ in range(repeats):
            X_perm = X_te.copy()
            rng.shuffle(X_perm[:, j])
            deltas.append(mse(X_perm, y_te, theta) - base)
        imp_mean[j] = np.mean(deltas)
        imp_std[j] = np.std(deltas)
    log(f"\n置换重要性（打乱特征后测试 MSE 增量，{repeats} 次平均）:")
    log(f"{'特征':<24}{'ΔMSE 均值':>14}{'标准差':>12}")
    for idx in np.argsort(-imp_mean):
        log(f"{names[idx]:<22}{imp_mean[idx]:>14.6f}{imp_std[idx]:>12.6f}")

    order = np.argsort(imp_mean)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(np.array(names)[order], imp_mean[order], xerr=imp_std[order], color="#55a868")
    ax.set_xlabel("打乱该特征后的测试 MSE 增量 ΔMSE")
    ax.set_title("特征置换重要性")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "permutation_importance.png", dpi=150)
    plt.close(fig)

    # ---- Pearson 相关热力图（训练集）----
    corr = np.corrcoef(X_tr, rowvar=False)
    fig, ax = plt.subplots(figsize=(9.2, 7.6))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(names, fontsize=8)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if abs(corr[i, j]) > 0.5 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("训练集特征 Pearson 相关系数")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "correlation_heatmap.png", dpi=150)
    plt.close(fig)

    off = corr.copy()
    np.fill_diagonal(off, 0)
    j, k = np.unravel_index(np.argmax(np.abs(off)), off.shape)
    log(f"\n相关性最强的一对特征: {names[j]} 与 {names[k]}，r = {corr[j, k]:.4f}")

    log("\n共线性对照实验（标准化特征，正规方程求解）:")
    log(f"{'模型':<38}{'系数_j':>12}{'系数_k':>12}{'训练 MSE':>12}{'测试 MSE':>12}")
    jj, kk = names[j], names[k]
    variants = [
        ("保留两者", list(range(len(names)))),
        (f"去掉 {jj}", [i for i in range(len(names)) if i != j]),
        (f"去掉 {kk}", [i for i in range(len(names)) if i != k]),
    ]
    for label, cols in variants:
        theta_v = fit_ols(X_tr[:, cols], y_tr)
        cj = theta_v[cols.index(j)] if j in cols else float("nan")
        ck = theta_v[cols.index(k)] if k in cols else float("nan")
        log(f"{label:<36}{cj:>12.4f}{ck:>12.4f}"
            f"{mse(X_tr[:, cols], y_tr, theta_v):>12.6f}{mse(X_te[:, cols], y_te, theta_v):>12.6f}")

    log("\n其余变量的系数变化（保留两者 -> 去掉其一）:")
    theta_drop_k = fit_ols(X_tr[:, [i for i in range(len(names)) if i != k]], y_tr)
    cols_drop_k = [i for i in range(len(names)) if i != k]
    log(f"{'特征':<26}{'完整模型':>12}{'去掉 ' + kk:>16}")
    for i in range(len(names)):
        if i == k:
            continue
        log(f"{names[i]:<24}{coef[i]:>12.4f}{theta_drop_k[cols_drop_k.index(i)]:>16.4f}")

    log("\n图已保存: reports/figs/feature_coefficients.png, permutation_importance.png, correlation_heatmap.png")
    log.close()


if __name__ == "__main__":
    main()
