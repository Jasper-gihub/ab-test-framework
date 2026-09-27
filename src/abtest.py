# -*- coding: utf-8 -*-
"""abtest.py —— 可复用的 A/B 测试分析工具。

包含：样本量/功效计算、SRM 校验、比例与均值检验、CUPED 方差缩减、多重比较校正。
全部只用 numpy / scipy 实现，便于在任何环境复现。

用法示例：
    from abtest import sample_size_proportion, srm_check, two_proportion_ztest
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


# ---------------------------------------------------------------- 样本量与功效
def sample_size_proportion(baseline: float, mde_abs: float,
                           alpha: float = 0.05, power: float = 0.80) -> int:
    """比例类指标（如转化率）的单组样本量。

    参数
    ----
    baseline : 基线转化率，如 0.10
    mde_abs  : 最小可检测的绝对提升，如 0.01（即 10% -> 11%）
    alpha    : 显著性水平（双侧）
    power    : 功效，通常 0.80

    公式：n = (z_{1-α/2} + z_{power})^2 * [p1(1-p1) + p2(1-p2)] / (p2-p1)^2
    """
    p1 = baseline
    p2 = baseline + mde_abs
    z_a = stats.norm.ppf(1 - alpha / 2)
    z_b = stats.norm.ppf(power)
    n = ((z_a + z_b) ** 2) * (p1 * (1 - p1) + p2 * (1 - p2)) / (p2 - p1) ** 2
    return int(np.ceil(n))


def sample_size_mean(std: float, mde_abs: float,
                     alpha: float = 0.05, power: float = 0.80) -> int:
    """连续类指标（如人均消费额）的单组样本量：n = 2 * (z_a + z_b)^2 * σ^2 / Δ^2。"""
    z_a = stats.norm.ppf(1 - alpha / 2)
    z_b = stats.norm.ppf(power)
    n = 2 * ((z_a + z_b) ** 2) * (std ** 2) / (mde_abs ** 2)
    return int(np.ceil(n))


# ---------------------------------------------------------------- SRM 校验
def srm_check(counts: list[int], expected_ratio: list[float] | None = None) -> dict:
    """样本比例失配校验（Sample Ratio Mismatch）。

    实验系统若分流正常，各组占比应接近设计比例；显著偏离往往意味着分流 bug，
    此时后续所有结论都不可信 —— 这是实验分析的第一步。
    """
    counts = np.asarray(counts, dtype=float)
    if expected_ratio is None:
        expected_ratio = [1 / len(counts)] * len(counts)
    expected_ratio = np.asarray(expected_ratio, dtype=float)
    expected_ratio = expected_ratio / expected_ratio.sum()

    total = counts.sum()
    expected = expected_ratio * total
    chi2 = float(((counts - expected) ** 2 / expected).sum())
    dof = len(counts) - 1
    p = float(stats.chi2.sf(chi2, dof))
    return {
        "counts": counts.astype(int).tolist(),
        "observed_ratio": [round(float(c / total), 4) for c in counts],
        "expected_ratio": [round(float(r), 4) for r in expected_ratio],
        "chi2": round(chi2, 4),
        "dof": dof,
        "p_value": p,
        "pass": bool(p >= 0.001),   # 业界常用阈值：p < 0.001 视为存在 SRM
    }


# ---------------------------------------------------------------- 假设检验
@dataclass
class TestResult:
    metric: str
    control: float
    treatment: float
    absolute_lift: float
    relative_lift: float
    ci_low: float
    ci_high: float
    p_value: float
    significant: bool

    def as_row(self) -> dict:
        return {
            "metric": self.metric,
            "control": round(self.control, 6),
            "treatment": round(self.treatment, 6),
            "abs_lift": round(self.absolute_lift, 6),
            "rel_lift": round(self.relative_lift, 4),
            "ci_low": round(self.ci_low, 6),
            "ci_high": round(self.ci_high, 6),
            "p_value": round(self.p_value, 6),
            "significant": self.significant,
        }


def two_proportion_ztest(x_c: int, n_c: int, x_t: int, n_t: int,
                         metric: str = "metric", alpha: float = 0.05) -> TestResult:
    """两组比例差异检验（转化率场景），并给出差异的置信区间。"""
    p_c, p_t = x_c / n_c, x_t / n_t
    diff = p_t - p_c
    # 原假设下的合并比例（用于检验）
    p_pool = (x_c + x_t) / (n_c + n_t)
    se_pool = np.sqrt(p_pool * (1 - p_pool) * (1 / n_c + 1 / n_t))
    z = diff / se_pool if se_pool > 0 else 0.0
    p_value = float(2 * (1 - stats.norm.cdf(abs(z))))
    # 置信区间用非合并标准误（更保守）
    se = np.sqrt(p_c * (1 - p_c) / n_c + p_t * (1 - p_t) / n_t)
    z_crit = stats.norm.ppf(1 - alpha / 2)
    return TestResult(
        metric=metric, control=p_c, treatment=p_t,
        absolute_lift=diff, relative_lift=(diff / p_c if p_c else float("nan")),
        ci_low=diff - z_crit * se, ci_high=diff + z_crit * se,
        p_value=p_value, significant=bool(p_value < alpha),
    )


def welch_ttest(a: np.ndarray, b: np.ndarray, metric: str = "metric",
                alpha: float = 0.05) -> TestResult:
    """连续指标的两组均值差异检验（Welch t 检验，方差不齐更稳健）。"""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    diff = b.mean() - a.mean()
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    t = diff / se if se > 0 else 0.0
    dof = (a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b)) ** 2 / (
        (a.var(ddof=1) / len(a)) ** 2 / (len(a) - 1) + (b.var(ddof=1) / len(b)) ** 2 / (len(b) - 1)
    )
    p_value = float(2 * (1 - stats.t.cdf(abs(t), dof)))
    z_crit = stats.t.ppf(1 - alpha / 2, dof)
    return TestResult(
        metric=metric, control=float(a.mean()), treatment=float(b.mean()),
        absolute_lift=diff, relative_lift=(diff / a.mean() if a.mean() else float("nan")),
        ci_low=diff - z_crit * se, ci_high=diff + z_crit * se,
        p_value=p_value, significant=bool(p_value < alpha),
    )


# ---------------------------------------------------------------- CUPED
def cuped_adjust(y: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, float]:
    """CUPED（利用实验前指标做协变量降低方差）。

    y_adj = y - θ * (x - mean(x))，其中 θ = Cov(y, x) / Var(x)。
    若 x 是实验前、与处理无关的指标，则调整后均值无偏且方差更小。
    返回 (调整后指标, θ)。
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    theta = float(np.cov(y, x, ddof=1)[0, 1] / np.var(x, ddof=1))
    return y - theta * (x - x.mean()), theta


def variance_reduction(y: np.ndarray, x: np.ndarray) -> float:
    """CUPED 带来的方差下降比例（越大越省样本）。"""
    _, theta = cuped_adjust(y, x)
    var_y = np.var(y, ddof=1)
    var_adj = np.var(y - theta * (x - np.mean(x)), ddof=1)
    return float(1 - var_adj / var_y)


# ---------------------------------------------------------------- 多重比较
def adjust_pvalues(p_values: list[float], method: str = "bh") -> list[float]:
    """多重比较校正。method: 'bonferroni' 或 'bh'（Benjamini-Hochberg, 控制 FDR）。"""
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    if method == "bonferroni":
        return [float(min(1.0, v * m)) for v in p]
    if method != "bh":
        raise ValueError("method 必须是 'bonferroni' 或 'bh'")
    order = np.argsort(p)
    ranked = p[order]
    adjusted = ranked * m / (np.arange(1, m + 1))
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]   # 保持单调
    out = np.empty(m)
    out[order] = np.clip(adjusted, 0, 1)
    return [float(v) for v in out]
