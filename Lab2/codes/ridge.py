"""实验二 高级：手写岭回归（L2 正则化，解析法）

约定：偏置项 b 不参与惩罚。实现上先对 X、y 去中心化，
      再求解 w = (Xc^T Xc + λI)^(-1) Xc^T yc，b = ȳ - x̄^T w，
      该解与“对 [X,1] 加惩罚掩码、不惩罚最后一维”的解析解等价。

- 5 折交叉验证在训练集内选择 λ
- 报告平均训练误差 / 平均验证误差，以及留出测试集误差
- 绘制 λ-MSE 曲线与系数收缩路径

运行：python codes/ridge.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split

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


def fit_ridge(X, y, lam):
    """解析法岭回归，偏置项不惩罚。"""
    xm, ym = X.mean(axis=0), y.mean()
    Xc, yc = X - xm, y - ym
    A = Xc.T @ Xc + lam * np.eye(X.shape[1])
    w = np.linalg.solve(A, Xc.T @ yc)
    b = ym - xm @ w
    return w, b


def predict(X, w, b):
    return X @ w + b


def mse(X, y, w, b):
    return float(np.mean((predict(X, w, b) - y) ** 2))


def main():
    log = Tee(ROOT / "reports" / "ridge_outputs.txt")
    names, X_tr, X_te, y_tr, y_te = load_split()
    lambdas = np.logspace(-4, 3, 22)

    log("=" * 70)
    log("实验二 高级：手写岭回归（解析法，偏置项不惩罚）")
    log("=" * 70)
    log(f"训练集 {len(y_tr)} 条，测试集 {len(y_te)} 条；λ 网格 {lambdas[0]:g} ~ {lambdas[-1]:g}（{len(lambdas)} 个）")

    # ---- 5 折交叉验证 ----
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_train = np.zeros((len(lambdas), 5))
    cv_val = np.zeros((len(lambdas), 5))
    for fi, (tr_idx, va_idx) in enumerate(kf.split(X_tr)):
        Xa, ya = X_tr[tr_idx], y_tr[tr_idx]
        Xv, yv = X_tr[va_idx], y_tr[va_idx]
        for li, lam in enumerate(lambdas):
            w, b = fit_ridge(Xa, ya, lam)
            cv_train[li, fi] = mse(Xa, ya, w, b)
            cv_val[li, fi] = mse(Xv, yv, w, b)

    val_mean = cv_val.mean(axis=1)
    best_i = int(np.argmin(val_mean))
    best_lam = lambdas[best_i]
    log("\n5 折交叉验证结果（每折的训练/验证 MSE）:")
    header = "λ".rjust(12) + "".join(f"{'折' + str(i + 1):>18}" for i in range(5)) + f"{'平均验证':>14}"
    log(header)
    for li in [0, len(lambdas) // 4, len(lambdas) // 2, 3 * len(lambdas) // 4, len(lambdas) - 1, best_i]:
        row = f"{lambdas[li]:>12.4g}"
        for fi in range(5):
            row += f"{cv_train[li, fi]:>8.4f}/{cv_val[li, fi]:<9.4f}"
        row += f"{val_mean[li]:>14.6f}"
        log(row)
    log("（单元格格式：训练 MSE / 验证 MSE）")
    log(f"\n最优 λ = {best_lam:.6g}")
    log(f"平均训练误差（5 折训练 MSE 均值）= {cv_train[best_i].mean():.6f}")
    log(f"平均验证误差（5 折验证 MSE 均值）= {cv_val[best_i].mean():.6f}")
    log(f"各折验证 MSE 标准差 = {cv_val[best_i].std():.6f}")

    # ---- 全训练集重训 + 测试集评估 ----
    w_best, b_best = fit_ridge(X_tr, y_tr, best_lam)
    train_mse = mse(X_tr, y_tr, w_best, b_best)
    test_mse = mse(X_te, y_te, w_best, b_best)
    log(f"\n用最优 λ 在全部训练集上重训：训练 MSE={train_mse:.6f}, 测试 MSE={test_mse:.6f}")

    w_ols, b_ols = fit_ridge(X_tr, y_tr, 0.0)
    log(f"λ=0（等价普通最小二乘）：训练 MSE={mse(X_tr, y_tr, w_ols, b_ols):.6f}, "
        f"测试 MSE={mse(X_te, y_te, w_ols, b_ols):.6f}")

    log("\n系数对比:")
    log(f"{'特征':<24}{'OLS (λ=0)':>14}{'Ridge (λ*)':>14}")
    for i, name in enumerate(names):
        log(f"{name:<22}{w_ols[i]:>14.4f}{w_best[i]:>14.4f}")
    log(f"{'偏置项':<22}{b_ols:>14.4f}{b_best:>14.4f}")
    log(f"{'||w||_2':<22}{np.linalg.norm(w_ols):>14.4f}{np.linalg.norm(w_best):>14.4f}")

    # ---- λ-MSE 曲线 ----
    test_curve = [mse(X_te, y_te, *fit_ridge(X_tr, y_tr, lam)) for lam in lambdas]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(lambdas, cv_train.mean(axis=1), label="训练 MSE（5 折均值）", marker="o", markersize=3)
    ax.plot(lambdas, val_mean, label="验证 MSE（5 折均值）", marker="s", markersize=3)
    ax.plot(lambdas, test_curve, label="测试 MSE", linestyle="--")
    ax.axvline(best_lam, color="gray", linestyle=":", label=f"λ*={best_lam:.3g}")
    ax.set_xscale("log")
    ax.set_xlabel("正则化系数 λ（对数刻度）")
    ax.set_ylabel("MSE")
    ax.set_title("岭回归：不同 λ 下的 MSE")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "ridge_lambda_mse.png", dpi=150)
    plt.close(fig)

    # ---- 系数收缩路径 ----
    paths = np.array([fit_ridge(X_tr, y_tr, lam)[0] for lam in lambdas])
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, name in enumerate(names):
        ax.plot(lambdas, paths[:, i], label=name, linewidth=1.3)
    ax.axvline(best_lam, color="gray", linestyle=":")
    ax.set_xscale("log")
    ax.set_xlabel("正则化系数 λ（对数刻度）")
    ax.set_ylabel("标准化回归系数")
    ax.set_title(f"岭回归系数收缩路径（虚线为 λ*={best_lam:.3g}）")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "ridge_coef_path.png", dpi=150)
    plt.close(fig)

    log("\n图已保存: reports/figs/ridge_lambda_mse.png, ridge_coef_path.png")
    log.close()


if __name__ == "__main__":
    main()
