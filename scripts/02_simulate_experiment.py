# -*- coding: utf-8 -*-
"""02_simulate_experiment.py —— 生成模拟实验数据（可复现）。

为什么用模拟数据：真实实验数据受保密约束无法公开；模拟数据可以**精确知道真实效应**，
从而验证分析流程能否把真效应检出来（方法论项目里最该做的事）。生成三份数据：
  1. ab_data.csv        正常分流、真实效应存在（转化率 +10%、GMV +4%）
  2. ab_data_srm.csv    分流异常（SRM），用于演示"先校验 SRM"的重要性
  3. ab_data_null.csv   真实效应为 0（负面对照），用于检验假阳性率

两个设计要点：
  * **显式随机分配**：先生成全部用户，再用洗牌把用户随机分配到两组（而不是分别生成两组），
    这样协变量在两组间才是平衡的 —— 真实实验同理。
  * **潜在消费能力 latent**：它同时驱动实验前消费（CUPED 协变量）与实验期行为。
    若协变量与结果指标无关，CUPED 就毫无作用；这是 CUPED 能生效的前提。

用法：python scripts/02_simulate_experiment.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

N_PER_GROUP = 32_000        # 由 01_power_analysis 得出：检测 10% 相对提升约需每组 3.2 万人
BASE_CVR = 0.05             # 基线转化率
TRUE_REL_LIFT = 0.10        # 真实效应：转化率相对 +10%
TRUE_GMV_LIFT = 0.04        # 真实效应：GMV +4%
TRUE_DURATION_LIFT = 0.03   # 真实效应：访问时长 +3%
SEED = 20260927


def _base_users(n: int, rng: np.random.Generator) -> pd.DataFrame:
    """生成用户基础属性与实验前消费（CUPED 协变量）。"""
    latent = rng.gamma(shape=3.0, scale=1.0, size=n)               # 潜在消费能力，均值 3
    # 实验前 30 天消费：由 latent 驱动，噪声相对小（与 latent 相关性更高，CUPED 才有用）
    pre_gmv = latent * rng.gamma(shape=6.0, scale=12.0, size=n)    # 均值≈216
    return pd.DataFrame({
        "latent": latent.round(4),
        "pre_gmv_30d": pre_gmv.round(2),
        "is_new_user": rng.random(n) < 0.35,
        "device": rng.choice(["iOS", "Android"], size=n, p=[0.45, 0.55]),
        "city_tier": rng.choice(["一线", "新一线", "二三线"], size=n, p=[0.2, 0.35, 0.45]),
    })


def _outcomes(df: pd.DataFrame, lift: float, rng: np.random.Generator) -> pd.DataFrame:
    """按给定真实效应生成结果指标。"""
    treatment = (df["group"] == "treatment").astype(float)

    p_conv = (BASE_CVR
              + BASE_CVR * lift * treatment
              + 0.0018 * df["latent"]
              - 0.012 * df["is_new_user"].astype(float))
    p_conv = np.clip(p_conv, 0.001, 0.95)
    converted = rng.random(len(df)) < p_conv

    duration = (rng.gamma(2.5, 40, size=len(df))
                * (0.55 + 0.35 * df["latent"])
                * (1 + TRUE_DURATION_LIFT * treatment))

    base_gmv = df["latent"] * rng.lognormal(mean=4.0, sigma=0.35, size=len(df))
    gmv = np.where(converted, base_gmv * (1 + TRUE_GMV_LIFT * treatment), 0.0)

    return pd.DataFrame({
        "converted": converted.astype(int),
        "gmv": gmv.round(2),
        "session_duration": duration.round(1),
        "clicks": rng.poisson(4.5, size=len(df)) + converted * 2,
    })


def _make_dataset(lift: float, treatment_ratio: float, seed: int) -> pd.DataFrame:
    """生成一份实验数据：先生成全部用户，再显式随机分配到两组。"""
    rng = np.random.default_rng(seed)
    n_t = int(N_PER_GROUP * treatment_ratio)
    n_total = N_PER_GROUP + n_t

    users = _base_users(n_total, rng)
    users["user_id"] = [f"u{seed % 1000}_{i:06d}" for i in range(n_total)]

    # 显式随机分配（洗牌）：保证协变量在两组间平衡
    idx = rng.permutation(n_total)
    group = np.array(["control"] * n_total, dtype=object)
    group[idx[:n_t]] = "treatment"
    users["group"] = group

    out = _outcomes(users, lift, rng)
    return pd.concat([users, out], axis=1)


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    normal = _make_dataset(lift=TRUE_REL_LIFT, treatment_ratio=1.0, seed=SEED)
    normal.to_csv(DATA_DIR / "ab_data.csv", index=False, encoding="utf-8-sig")

    srm = _make_dataset(lift=TRUE_REL_LIFT, treatment_ratio=0.97, seed=SEED + 1)
    srm.to_csv(DATA_DIR / "ab_data_srm.csv", index=False, encoding="utf-8-sig")

    null = _make_dataset(lift=0.0, treatment_ratio=1.0, seed=SEED + 2)
    null.to_csv(DATA_DIR / "ab_data_null.csv", index=False, encoding="utf-8-sig")

    for name, d in [("ab_data", normal), ("ab_data_srm", srm), ("ab_data_null", null)]:
        print(f"{name}.csv  行数={len(d):,}  "
              f"control={int((d['group'] == 'control').sum()):,} "
              f"treatment={int((d['group'] == 'treatment').sum()):,}  "
              f"转化率={d['converted'].mean():.4f}  人均GMV={d['gmv'].mean():.2f}")

    corr = np.corrcoef(normal["pre_gmv_30d"], normal["gmv"])[0, 1]
    corr_d = np.corrcoef(normal["pre_gmv_30d"], normal["session_duration"])[0, 1]
    print(f"\n协变量相关性：pre_gmv_30d vs GMV = {corr:.3f}；vs 访问时长 = {corr_d:.3f}")
    print("（相关性越强，CUPED 的方差下降越多；零膨胀指标如人均 GMV 的相关性天然偏低）")
    print(f"\n真实效应：转化率 +{TRUE_REL_LIFT:.0%}、GMV +{TRUE_GMV_LIFT:.0%}、"
          f"时长 +{TRUE_DURATION_LIFT:.0%}；负面对照为 0；随机种子 {SEED}（可复现）")


if __name__ == "__main__":
    main()
