# src/online_ts_forecast/traditional/run_layer0_chiller.py

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from online_ts_forecast.traditional.baselines import (
    naive_forecast,
    seasonal_naive_forecast,
    moving_average_forecast,
    ets_forecast,
    rolling_backtest,
)
from online_ts_forecast.traditional.metrics import evaluate_backtest


# ======= 配置区域（你可以之后改成读取 yaml 配置） =======
WARMUP_DAYS = 7
EVAL_DAYS = 7

DATA_PATH = Path("data/private/tem_data_07.csv")
TARGET_COL = "冷机3总有功功率"  # 可以先选一个你最熟悉的指标
FREQ_MINUTES = 5
DAILY_SEASON = int(24 * 60 / FREQ_MINUTES)  # 288
HORIZON = 1  # 先做 +5min 单步预测
MIN_HISTORY =  max(DAILY_SEASON * 2, WARMUP_DAYS * DAILY_SEASON)  # 至少两天作为训练历史


OUTPUT_METRICS_DIR = Path("outputs/metrics")
OUTPUT_FIGURES_DIR = Path("outputs/figures")


def load_chiller_series() -> pd.Series:
    df = pd.read_csv(DATA_PATH)
    time_col = df.columns[0]
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values(time_col)

    # 设置时间索引
    df = df.set_index(time_col)

    # 只取单一目标列
    s = pd.to_numeric(df[TARGET_COL], errors="coerce").dropna()
    return s


def plot_backtest(df: pd.DataFrame, title: str, out_path: Path) -> None:
    if df.empty:
        print(f"[WARN] 空回测结果，跳过绘图：{title}")
        return

    plt.figure(figsize=(12, 4))
    t = df["time"]
    y = df["y_true"]
    yhat = df["y_pred"]

    plt.plot(t, y, label="真实", linewidth=1.2)
    plt.plot(t, yhat, label="预测", linewidth=1.0)

    plt.title(title)
    plt.xlabel("时间")
    plt.ylabel(TARGET_COL)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[图] 已保存：{out_path}")


def run_all_baselines():
    series = load_chiller_series()
    
    # 以测试集开头为基准：前 7 天预热，接下来 7 天评估
    t0 = series.index.min()
    eval_start = t0 + pd.Timedelta(days=WARMUP_DAYS)
    eval_end = eval_start + pd.Timedelta(days=EVAL_DAYS)

    # 防御：数据不够长时截断
    tmax = series.index.max()
    if eval_start >= tmax:
        raise ValueError(f"数据不足：eval_start={eval_start} 已超出数据末尾 {tmax}")
    if eval_end > tmax:
        eval_end = tmax
    OUTPUT_METRICS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    methods = {
        "naive": (naive_forecast, {}),
        "seasonal_naive": (seasonal_naive_forecast, {"season_length": DAILY_SEASON}),
        "moving_average": (moving_average_forecast, {"window": 12}),  # 最近1小时
        "ets": (
            ets_forecast,
            {"seasonal": "add", "season_length": DAILY_SEASON, "trend": "add"},
        ),
    }

    metrics_rows = []

    for name, (func, kwargs) in methods.items():
        print(f"\n=== 运行基线：{name} ===")
        df_bt = rolling_backtest(
            series,
            forecast_func=func,
            horizon=HORIZON,
            min_history=MIN_HISTORY,
            forecast_kwargs=kwargs,
            start_time=eval_start,
            end_time=eval_end,
        )

        tag = f"{eval_start:%Y%m%d}_{eval_end:%Y%m%d}"
        # 保存回测明细
        out_csv = OUTPUT_METRICS_DIR / f"layer0_{name}_chiller_h{HORIZON}.csv"
        df_bt.to_csv(out_csv, index=False, encoding="utf-8-sig")
        print(f"[CSV] 已保存回测结果：{out_csv}，样本数={len(df_bt)}")

        # 计算指标
        m = evaluate_backtest(df_bt)
        m["method"] = name
        metrics_rows.append(m)

        # 画图
        out_png = OUTPUT_FIGURES_DIR / f"layer0_{name}_chiller_h{HORIZON}.png"
        plot_backtest(
            df_bt,
            title=f"{TARGET_COL} - {name} 基线回测 (h={HORIZON})",
            out_path=out_png,
        )
        

    # 汇总 metrics
    if metrics_rows:
        df_metrics = pd.DataFrame(metrics_rows)
        df_metrics = df_metrics[["method", "MAE", "RMSE", "sMAPE", "n"]]
        summary_csv = OUTPUT_METRICS_DIR / f"layer0_summary_chiller_h{HORIZON}.csv"
        df_metrics.to_csv(summary_csv, index=False, encoding="utf-8-sig")
        print("\n=== Layer 0 基线表现 ===")
        print(df_metrics)
        print(f"[汇总] 已保存：{summary_csv}")


if __name__ == "__main__":
    run_all_baselines()
