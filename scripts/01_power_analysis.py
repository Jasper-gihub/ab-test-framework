# -*- coding: utf-8 -*-
"""01_power_analysis.py —— 实验前的样本量与功效计算。

场景：某电商 App 测试新版推荐算法，主指标是下单转化率（基线 5%）。
问题：要在 80% 功效、5% 显著性下检测出不同幅度的提升，需要多少样本？

用法：python scripts/01_power_analysis.py
产出：reports/figures/power_curve.png、控制台表格
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from abtest import sample_size_mean, sample_size_proportion  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "reports" / "figures"

BASELINE_CVR = 0.05          # 基线转化率 5%
BASELINE_GMV_STD = 60.0      # 人均消费额标准差（元）


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print(f"基线转化率 {BASELINE_CVR:.1%} ｜ 显著性 5% ｜ 功效 80%\n")
    print("一、比例指标（转化率）：不同相对提升所需样本量")
    print(f"{'相对提升':>8} {'绝对提升':>10} {'每组样本':>10} {'总样本':>10} {'预计实验周期*':>12}")
    rows = []
    for rel in [0.05, 0.10, 0.15, 0.20, 0.30]:
        mde = BASELINE_CVR * rel
        n = sample_size_proportion(BASELINE_CVR, mde)
        # 假设日活 50,000、50/50 分流
        days = np.ceil(n / 25000)
        rows.append((rel, mde, n, n * 2, days))
        print(f"{rel:>7.0%} {mde:>10.4f} {n:>10,} {n * 2:>10,} {days:>10.0f} 天")
    print("* 按日活 5 万、50/50 分流估算（只算单指标所需的最小样本量）\n")

    print("二、连续指标（人均消费额，σ=60 元）：不同绝对提升所需样本量")
    print(f"{'绝对提升':>10} {'相对提升*':>10} {'每组样本':>10} {'总样本':>10}")
    for mde in [2.0, 5.0, 10.0]:
        n = sample_size_mean(BASELINE_GMV_STD, mde)
        # 参考：人均消费额基线约 160 元
        print(f"{mde:>10.1f} {mde / 160:>10.1%} {n:>10,} {n * 2:>10,}")
    print("* 分母取人均消费额基线 160 元\n")

    # ---- 功效曲线 ----
    rels = np.linspace(0.02, 0.40, 40)
    ns = [sample_size_proportion(BASELINE_CVR, BASELINE_CVR * r) for r in rels]
    fig, ax = plt.subplots(figsize=(8, 4.6), dpi=160)
    ax.plot(rels * 100, ns, lw=2, color="#1F6FB2")
    ax.set_xlabel("相对提升（%）")
    ax.set_ylabel("每组所需样本量")
    ax.set_title("转化率实验：所需样本量 vs 最小可检测提升（基线 5%，α=0.05，power=0.8）",
                 fontsize=11)
    ax.grid(alpha=0.25)
    for r, n in [(0.10, sample_size_proportion(BASELINE_CVR, BASELINE_CVR * 0.10)),
                 (0.20, sample_size_proportion(BASELINE_CVR, BASELINE_CVR * 0.20))]:
        ax.scatter([r * 100], [n], color="#d9534f", zorder=5)
        ax.annotate(f"{r:.0%} → {n:,}", xy=(r * 100, n), xytext=(r * 100 + 2, n * 1.05),
                    fontsize=9, color="#a33")
    fig.tight_layout()
    out = FIG_DIR / "power_curve.png"
    fig.savefig(out)
    print(f"图 -> {out}")

    print("结论：想检测 10% 的相对提升（5%→5.5%），每组需要约 "
          f"{sample_size_proportion(BASELINE_CVR, BASELINE_CVR * 0.10):,} 人；"
          "检测 5% 的提升则要 4 倍以上样本 —— 这就是"
          "“小提升需要大样本”的量化体现，也是很多实验只能检出大效应的原因。")


if __name__ == "__main__":
    main()
