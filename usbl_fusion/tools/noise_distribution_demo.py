#!/usr/bin/env python3
"""
高斯 (Gaussian) 噪声 vs 拉普拉斯 (Laplace) 噪声 — 说明与可视化

数学摘要（零均值、一维）：
- **高斯** X ~ N(0, σ²)：密度 f(x) ∝ exp(-x²/(2σ²))，|x| 很大时概率按 **exp(-x²)** 衰减，**尾薄**。
- **拉普拉斯** X ~ Laplace(0, b)（与 numpy `laplace(loc=0, scale=b)` 一致）：密度 f(x) ∝ exp(-|x|/b)，
  方差 **Var(X) = 2b²**。|x| 大时按 **exp(-|x|)** 衰减，相对同方差的高斯 **更容易出现较大偏差（重尾）**。

对比方式：
- 给定高斯标准差 σ，取拉普拉斯 scale **b = σ/√2**，使 **Var_Laplace = 2b² = σ² = Var_Gaussian**，
  即 **同方差** 比较（公平对比一阶离散程度）。
- 直方图 + 理论密度曲线；可选对数纵坐标以观察尾部。

用法（在 usbl_fusion 目录下）：
  .venv/bin/python tools/noise_distribution_demo.py
  .venv/bin/python tools/noise_distribution_demo.py --sigma 0.8 -n 80000 --out outputs/plots_tools/noise_compare.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _gaussian_pdf(x: np.ndarray, sigma: float) -> np.ndarray:
    s = max(float(sigma), 1e-12)
    return (1.0 / (s * np.sqrt(2.0 * np.pi))) * np.exp(-0.5 * (x / s) ** 2)


def _laplace_pdf(x: np.ndarray, b: float) -> np.ndarray:
    b = max(float(b), 1e-12)
    return (1.0 / (2.0 * b)) * np.exp(-np.abs(x) / b)


def main() -> None:
    parser = argparse.ArgumentParser(description="Gaussian vs Laplace 噪声说明与绘图")
    parser.add_argument(
        "--sigma",
        type=float,
        default=0.8,
        help="高斯标准差 σ (m)，与 marine.ini 中 usbl_burst_scale 量级可对照",
    )
    parser.add_argument(
        "-n",
        "--n-samples",
        type=int,
        default=100_000,
        dest="n_samples",
        help="蒙特卡洛样本数",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="",
        help="输出 PNG 路径（默认 outputs/plots_tools/noise_gs_vs_laplace.png）",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--xmax",
        type=float,
        default=4.0,
        help="直方图与密度图的横轴范围 [-xmax, xmax]（单位与 sigma 一致）",
    )
    args = parser.parse_args()

    sigma = float(args.sigma)
    b = sigma / np.sqrt(2.0)  # 同方差：2 b^2 = sigma^2
    rng = np.random.default_rng(int(args.seed))
    n = int(max(1000, args.n_samples))

    gs = rng.normal(0.0, sigma, size=n)
    lap = rng.laplace(0.0, b, size=n)

    root = Path(__file__).resolve().parent.parent
    out = Path(args.out) if args.out else root / "outputs" / "plots_tools" / "noise_gs_vs_laplace.png"
    out.parent.mkdir(parents=True, exist_ok=True)

    xmax = float(max(args.xmax, 3.0 * sigma))
    bins = min(200, max(60, int(np.sqrt(n))))
    xs = np.linspace(-xmax, xmax, 800)

    fig, axes = plt.subplots(2, 2, figsize=(10.0, 8.0))

    ax = axes[0, 0]
    ax.hist(gs, bins=bins, range=(-xmax, xmax), density=True, alpha=0.55, color="tab:blue", label="MC Gaussian")
    ax.plot(xs, _gaussian_pdf(xs, sigma), "k-", lw=2.0, label=rf"pdf $N(0,\sigma^2)$, $\sigma$={sigma:g}")
    ax.set_title(r"Gaussian $N(0,\sigma^2)$ — thin tails")
    ax.set_xlabel("x")
    ax.set_ylabel("density")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.hist(lap, bins=bins, range=(-xmax, xmax), density=True, alpha=0.55, color="tab:orange", label="MC Laplace")
    ax.plot(xs, _laplace_pdf(xs, b), "k-", lw=2.0, label=rf"pdf Laplace$(0,b)$, $b$={b:.4g}")
    ax.set_title(r"Laplace$(0,b)$ with $b=\sigma/\sqrt{2}$ so Var$=\sigma^2$")
    ax.set_xlabel("x")
    ax.set_ylabel("density")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.hist(gs, bins=bins, range=(-xmax, xmax), density=True, alpha=0.45, color="tab:blue", label="Gaussian")
    ax.hist(lap, bins=bins, range=(-xmax, xmax), density=True, alpha=0.45, color="tab:orange", label="Laplace")
    ax.plot(xs, _gaussian_pdf(xs, sigma), "b-", lw=1.8)
    ax.plot(xs, _laplace_pdf(xs, b), color="darkorange", lw=1.8)
    ax.set_title("Equal variance: histograms + pdf")
    ax.set_xlabel("x")
    ax.set_ylabel("density")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    thr = np.linspace(2.0 * sigma, xmax, 50)
    p_gs = np.array([np.mean(np.abs(gs) > t) for t in thr])
    p_lap = np.array([np.mean(np.abs(lap) > t) for t in thr])
    ax.semilogy(thr, np.maximum(p_gs, 1e-6), "b.-", ms=4, label=r"$P(|X|>t)$ Gaussian")
    ax.semilogy(thr, np.maximum(p_lap, 1e-6), color="darkorange", marker=".", ls="-", ms=4, label=r"$P(|X|>t)$ Laplace")
    ax.set_xlabel("threshold t")
    ax.set_ylabel(r"$P(|X|>t)$ (log)")
    ax.set_title("Tails: Laplace exceeds large |x| more often (same Var)")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)

    fig.suptitle(
        rf"Gaussian vs Laplace (equal variance): $\sigma$={sigma:g}, $b=\sigma/\sqrt{{2}}$, $n$={n}",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out.resolve()}")


if __name__ == "__main__":
    main()
