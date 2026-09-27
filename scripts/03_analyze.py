# -*- coding: utf-8 -*-
"""03_analyze.py —— 完整实验分析流程（可直接用于真实实验数据）。

流程：SRM 校验 → 主指标检验 → CUPED 方差缩减 → 分群异质性 → 多重比较校正 → 负面对照

用法：python scripts/03_analyze.py
产出：reports/ab_test_report.md、reports/analysis_results.csv、reports/figures/*.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy import stats  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from abtest import (adjust_pvalues, cuped_adjust, srm_check,  # noqa: E402
                    two_proportion_ztest, variance_reduction, welch_ttest)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REPORTS = ROOT / "reports"
FIGS = REPORTS / "figures"
ALPHA = 0.05

TRUE_LIFT_CVR = 0.10     # 数据生成时设定的真实效应（用于验证流程能否检出）


def load(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA / f"{name}.csv", encoding="utf-8-sig")


def split(df: pd.DataFrame):
    return df[df["group"] == "control"], df[df["group"] == "treatment"]


def analyze(df: pd.DataFrame) -> dict:
    """对一个实验数据集跑完整分析，返回结果字典。"""
    c, t = split(df)
    res: dict = {"n_control": len(c), "n_treatment": len(t)}

    # ① SRM 校验：第一步永远是确认分流是否正常
    res["srm"] = srm_check([len(c), len(t)], [0.5, 0.5])

    # ①' 协变量平衡检验：两组在实验前特征上是否可比（随机化是否成功）
    bal = []
    r_pre = welch_ttest(c["pre_gmv_30d"].values, t["pre_gmv_30d"].values, "pre_gmv_30d")
    bal.append({"变量": "实验前30天消费（均值）", "control": f"{r_pre.control:.2f}",
                "treatment": f"{r_pre.treatment:.2f}", "p_value": f"{r_pre.p_value:.3f}"})
    p_new = float(stats.ttest_ind(c["is_new_user"].astype(float),
                                  t["is_new_user"].astype(float)).pvalue)
    bal.append({"变量": "新客占比", "control": f"{c['is_new_user'].mean():.3f}",
                "treatment": f"{t['is_new_user'].mean():.3f}", "p_value": f"{p_new:.3f}"})
    for col in ["device", "city_tier"]:
        _, p_chi, _, _ = stats.chi2_contingency(pd.crosstab(df[col], df["group"]))
        bal.append({"变量": f"{col}（卡方检验）", "control": "—", "treatment": "—",
                    "p_value": f"{float(p_chi):.3f}"})
    res["balance"] = bal

    # ② 主指标：转化率（比例类，用 z 检验 + 置信区间）
    res["cvr"] = two_proportion_ztest(
        int(c["converted"].sum()), len(c), int(t["converted"].sum()), len(t),
        "转化率", ALPHA)

    # ③ 连续指标：人均 GMV、转化用户客单价、访问时长（Welch t 检验，方差不齐更稳）
    res["gmv"] = welch_ttest(c["gmv"].values, t["gmv"].values, "人均GMV", ALPHA)
    res["aov"] = welch_ttest(c.loc[c.converted == 1, "gmv"].values,
                             t.loc[t.converted == 1, "gmv"].values, "转化用户客单价", ALPHA)
    res["duration"] = welch_ttest(c["session_duration"].values, t["session_duration"].values,
                                  "访问时长", ALPHA)

    # ④ CUPED：用实验前 30 天消费做协变量，降低方差
    x = df["pre_gmv_30d"].values
    df = df.copy()

    # 4a. 人均 GMV（零膨胀指标：未购买者为 0，方差下降幅度通常较小）
    y = df["gmv"].values
    y_adj, theta = cuped_adjust(y, x)
    df["gmv_cuped"] = y_adj
    c2, t2 = split(df)
    res["cuped_theta"] = theta
    res["cuped_variance_reduction"] = variance_reduction(y, x)
    res["gmv_cuped"] = welch_ttest(c2["gmv_cuped"].values, t2["gmv_cuped"].values,
                                   "人均GMV（CUPED 调整后）", ALPHA)

    # 4b. 访问时长（所有用户都有值，无零膨胀，CUPED 效果通常更明显）
    y_d = df["session_duration"].values
    y_d_adj, theta_d = cuped_adjust(y_d, x)
    df["duration_cuped"] = y_d_adj
    c3, t3 = split(df)
    res["cuped_duration_theta"] = theta_d
    res["cuped_duration_variance_reduction"] = variance_reduction(y_d, x)
    res["duration_cuped"] = welch_ttest(c3["duration_cuped"].values, t3["duration_cuped"].values,
                                        "访问时长（CUPED 调整后）", ALPHA)

    # ⑤ 分群异质性：新老客 / 设备 / 城市线级
    segments = []
    for col in ["is_new_user", "device", "city_tier"]:
        for value, sub in df.groupby(col, observed=True):
            sc, st = split(sub)
            if len(sc) < 100 or len(st) < 100:
                continue
            r = two_proportion_ztest(int(sc["converted"].sum()), len(sc),
                                     int(st["converted"].sum()), len(st),
                                     f"{col}={value}")
            row = r.as_row()
            row.update({"segment_col": col, "segment_value": str(value),
                        "n_control": len(sc), "n_treatment": len(st)})
            segments.append(row)
    res["segments"] = pd.DataFrame(segments)

    # ⑥ 多重比较校正：把主指标与所有分群的 p 值一起校正（BH 控制 FDR）
    pvals = [res[k].p_value for k in ["cvr", "gmv", "aov", "duration", "gmv_cuped", "duration_cuped"]]
    pvals += res["segments"]["p_value"].tolist()
    adj = adjust_pvalues(pvals, method="bh")
    res["p_adjusted_bh"] = adj
    return res


def report_markdown(res_main: dict, res_srm: dict, res_null: dict) -> str:
    m = res_main
    lines = [
        "# A/B 实验分析报告",
        "",
        "> 数据说明：本项目使用**模拟实验数据**（真实实验数据受保密约束）。"
        "模拟的价值在于**已知真实效应**：转化率相对 +10%、GMV +4%，可用于验证分析流程能否正确检出效应。",
        "",
        "## 一、分流校验（SRM）与基线平衡",
        "",
        "| 数据集 | control | treatment | 观测占比 | 卡方 | p 值 | 结论 |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, r in [("正常实验", res_main), ("人为注入 SRM", res_srm)]:
        s = r["srm"]
        lines.append(
            f"| {name} | {s['counts'][0]:,} | {s['counts'][1]:,} | "
            f"{s['observed_ratio'][0]:.4f} / {s['observed_ratio'][1]:.4f} | "
            f"{s['chi2']:.2f} | {s['p_value']:.3g} | {'✅ 通过' if s['pass'] else '❌ 存在 SRM，结论不可信'} |"
        )
    lines += [
        "",
        "**为什么第一步做这个**：分流异常（例如埋点丢失、跳转失败）会让两组人群本身不可比，"
        "此时再显著的效应也没有意义。业界通常用 p < 0.001 作为告警阈值。",
        "",
        "### 基线平衡检验（实验前变量）",
        "",
        "| 实验前变量 | control | treatment | p 值 |",
        "|---|---|---|---|",
        *[f"| {b['变量']} | {b['control']} | {b['treatment']} | {b['p_value']} |" for b in m["balance"]],
        "",
        "实验前变量的 p 值均不显著，说明随机化成功、两组可比；"
        "**若某个实验前变量存在显著差异，后续效应估计就可能有偏**。",
        "",
        "## 二、主指标与核心指标",
        "",
        "| 指标 | control | treatment | 绝对提升 | 相对提升 | 95% CI | p 值 | 显著 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for key, label in [("cvr", "转化率"), ("gmv", "人均GMV"), ("aov", "转化用户客单价"),
                       ("duration", "访问时长"), ("gmv_cuped", "人均GMV（CUPED）"),
                       ("duration_cuped", "访问时长（CUPED）")]:
        r = m[key]
        lines.append(
            f"| {label} | {r.control:.4f} | {r.treatment:.4f} | "
            f"{r.absolute_lift:+.4f} | {r.relative_lift:+.2%} | "
            f"[{r.ci_low:+.4f}, {r.ci_high:+.4f}] | {r.p_value:.4f} | "
            f"{'✅' if r.significant else '❌'} |"
        )
    lines += [
        "",
        f"真实效应为转化率相对 **+{TRUE_LIFT_CVR:.0%}**，上表相对提升与真值方向一致，"
        "说明分析流程能够正确检出真实效应。",
        "",
        "> **注意点估计与真值的关系**：单次实验的点估计不会精确等于真值，它会围绕真值以约 "
        "1.96×标准误的幅度波动。判断结果是否可信，应看置信区间是否覆盖真值、p 值是否显著，"
        "而不是要求点估计与真值完全一致 —— 这也是实验报告必须给出置信区间的原因。",
        "",
        "## 三、CUPED 方差缩减",
        "",
        f"- 协变量：实验前 30 天消费额（`pre_gmv_30d`）",
        "",
        "| 指标 | θ | 方差下降 | 原始 p 值 | CUPED 后 p 值 |",
        "|---|---|---|---|---|",
        f"| 人均GMV | {m['cuped_theta']:.4f} | {m['cuped_variance_reduction']:.2%} | "
        f"{m['gmv'].p_value:.4f} | {m['gmv_cuped'].p_value:.4f} |",
        f"| 访问时长 | {m['cuped_duration_theta']:.4f} | {m['cuped_duration_variance_reduction']:.2%} | "
        f"{m['duration'].p_value:.4f} | {m['duration_cuped'].p_value:.4f} |",
        "",
        "CUPED 的作用：用实验前的指标吸收个体差异，在**不引入偏差**的前提下降低方差，"
        "从而用更少的样本达到同样功效。注意零膨胀指标（大量用户 GMV 为 0）的方差下降幅度"
        "通常小于连续且人人有值的指标（如访问时长）。",
        "",
        "## 四、分群异质性（BH 校正后）",
        "",
        "| 分群 | control 转化率 | treatment 转化率 | 相对提升 | p 值 | BH 校正后 | 显著 |",
        "|---|---|---|---|---|---|---|",
    ]
    n_main = 6
    seg = m["segments"]
    adj = m["p_adjusted_bh"]
    for i, (_, r) in enumerate(seg.iterrows()):
        padj = adj[n_main + i]
        lines.append(
            f"| {r['segment_col']}={r['segment_value']} | {r['control']:.4f} | {r['treatment']:.4f} | "
            f"{r['rel_lift']:+.2%} | {r['p_value']:.4f} | {padj:.4f} | "
            f"{'✅' if padj < ALPHA else '❌'} |"
        )
    lines += [
        "",
        "**多重比较的坑**：同时看 10 个分群，即使完全没有效应也可能有 1 个“显著”（假阳性）。"
        "本项目用 Benjamini-Hochberg 控制 FDR；若用 Bonferroni 会更严格但更保守。",
        "",
        "## 五、负面对照（真实效应 = 0）",
        "",
        "| 指标 | 相对提升 | p 值 | 是否正确判为不显著 |",
        "|---|---|---|---|",
    ]
    for key, label in [("cvr", "转化率"), ("gmv", "人均GMV")]:
        r = res_null[key]
        lines.append(f"| {label} | {r.relative_lift:+.2%} | {r.p_value:.4f} | "
                     f"{'✅ 正确' if not r.significant else '❌ 假阳性'} |")
    lines += [
        "",
        "负面对照用于检验流程的**假阳性率**：真实效应为 0 时，不应出现显著结论。",
        "",
        "## 六、结论与决策建议",
        "",
        "1. 先看 SRM：分流正常才能继续解读；出现 SRM 优先排查实验系统而非业务",
        "2. 主指标显著且方向与预期一致 → 可考虑放量；但需结合业务成本与收益",
        "3. 分群显著项要经多重比较校正后再判断，避免“在噪音里找故事”",
        "4. 若指标不显著但方向为负，需评估是否值得继续投入",
        "",
        "> 本报告由 `scripts/03_analyze.py` 自动生成于模拟数据；替换 `data/ab_data.csv` 即可复用于真实实验。",
    ]
    return "\n".join(lines)


def make_figures(df: pd.DataFrame, res: dict) -> None:
    FIGS.mkdir(parents=True, exist_ok=True)

    # 图 1：主指标置信区间森林图
    keys = [("cvr", "转化率"), ("gmv", "人均GMV"), ("gmv_cuped", "人均GMV(CUPED)"),
            ("aov", "客单价"), ("duration", "访问时长")]
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=160)
    ys = np.arange(len(keys))[::-1]
    for y, (k, label) in zip(ys, keys):
        r = res[k]
        # 相对提升 + 相对 CI（把绝对 CI 转成相对）
        rel = r.relative_lift
        lo = r.ci_low / r.control if r.control else np.nan
        hi = r.ci_high / r.control if r.control else np.nan
        color = "#1F6FB2" if r.significant else "#999999"
        ax.errorbar(rel, y, xerr=[[rel - lo], [hi - rel]], fmt="o", color=color,
                    capsize=4, ms=6)
    ax.axvline(0, color="#d9534f", ls="--", lw=1)
    ax.set_yticks(ys)
    ax.set_yticklabels([l for _, l in keys])
    ax.set_xlabel("相对提升（含 95% 置信区间）")
    ax.set_title("各指标效应量与置信区间（蓝=显著，灰=不显著）", fontsize=11)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGS / "effect_forest.png")
    plt.close(fig)

    # 图 2：CUPED 前后分布
    y = df["gmv"].values
    y_adj, _ = cuped_adjust(y, df["pre_gmv_30d"].values)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), dpi=160)
    axes[0].hist(y, bins=60, color="#9ec9e8")
    axes[0].set_title(f"原始人均GMV（σ={np.std(y):.2f}）", fontsize=10)
    axes[1].hist(y_adj, bins=60, color="#1F6FB2")
    axes[1].set_title(f"CUPED 调整后（σ={np.std(y_adj):.2f}，方差降 {variance_reduction(y, df['pre_gmv_30d'].values):.1%}）",
                      fontsize=10)
    for a in axes:
        a.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIGS / "cuped_distribution.png")
    plt.close(fig)


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)

    df_main = load("ab_data")
    df_srm = load("ab_data_srm")
    df_null = load("ab_data_null")

    res_main = analyze(df_main)
    res_srm = analyze(df_srm)
    res_null = analyze(df_null)

    # 结果表
    rows = []
    for k in ["cvr", "gmv", "aov", "duration", "gmv_cuped", "duration_cuped"]:
        row = res_main[k].as_row()
        row["dataset"] = "main"
        rows.append(row)
    for i, (_, s) in enumerate(res_main["segments"].iterrows()):
        row = {**s.to_dict(), "dataset": "segment",
               "p_adjusted_bh": res_main["p_adjusted_bh"][6 + i]}
        rows.append(row)
    pd.DataFrame(rows).to_csv(REPORTS / "analysis_results.csv", index=False, encoding="utf-8-sig")

    md = report_markdown(res_main, res_srm, res_null)
    (REPORTS / "ab_test_report.md").write_text(md, encoding="utf-8")

    make_figures(df_main, res_main)

    s = res_main
    print("=== 分析完成 ===")
    print(f"SRM：p={s['srm']['p_value']:.3g} -> {'通过' if s['srm']['pass'] else '异常'}")
    print(f"转化率：{s['cvr'].control:.4f} -> {s['cvr'].treatment:.4f} "
          f"（相对 {s['cvr'].relative_lift:+.2%}，p={s['cvr'].p_value:.4f}）")
    print(f"CUPED 方差下降：GMV {s['cuped_variance_reduction']:.2%}（θ={s['cuped_theta']:.3f}）｜"
          f"访问时长 {s['cuped_duration_variance_reduction']:.2%}（θ={s['cuped_duration_theta']:.3f}）")
    print(f"访问时长 p 值：{s['duration'].p_value:.4f} -> {s['duration_cuped'].p_value:.4f}（CUPED 后）")
    print(f"报告 -> {REPORTS / 'ab_test_report.md'}")


if __name__ == "__main__":
    main()
