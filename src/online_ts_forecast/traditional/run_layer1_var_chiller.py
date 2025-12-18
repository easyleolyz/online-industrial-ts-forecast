# src/online_ts_forecast/traditional/run_layer1_var_chiller.py

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from online_ts_forecast.traditional.metrics import evaluate_backtest
from online_ts_forecast.traditional.var_models import rolling_backtest_var
from online_ts_forecast.utils.timing import time_block, attach_timing


# ======= 配置 =======

DATA_PATH = Path("data/private/tem_data_07.csv")
TIME_COL_INDEX = 0

TARGET_COL = "冷机3总有功功率"
COVARIATE_COLS = [
    "冷机3总有功功率",
    "冷机4总有功功率",
]

FREQ_MINUTES = 5
DAILY_SEASON = int(24 * 60 / FREQ_MINUTES)

HORIZON = 1
MIN_HISTORY = DAILY_SEASON * 2
MAXLAGS = 12

OUTPUT_METRICS_DIR = Path("outputs/metrics")
OUTPUT_FIGURES_DIR = Path("outputs/figures")


def load_chiller_multivar() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    time_col = df.columns[TIME_COL_INDEX]
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values(time_col).set_index(time_col)

    for c in COVARIATE_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df_cov = df[COVARIATE_COLS].dropna()
    return df_cov


def plot_backtest(df: pd.DataFrame, title: str, out_path: Path) -> None:
    if df.empty:
        print(f"[WARN] 空回测结果，跳过绘图：{title}")
        return

    t = df["time"]
    y = df["y_true"]
    yhat = df["y_pred"]

    plt.figure(figsize=(12, 4))
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


def run_layer1_var():
    timing_global = {}
    with time_block("load_chiller_multivar", timing_global):
        df_multi = load_chiller_multivar()

    OUTPUT_METRICS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("\n=== Layer1 VAR 多变量离线回测 ===")
    timing = {}
    with time_block("VAR.rolling_backtest", timing):
        df_bt = rolling_backtest_var(
            df_multi,
            target_col=TARGET_COL,
            horizon=HORIZON,
            min_history=MIN_HISTORY,
            maxlags=MAXLAGS,
        )

    out_csv = OUTPUT_METRICS_DIR / f"layer1_var_chiller_h{HORIZON}.csv"
    df_bt.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"[CSV] 已保存：{out_csv}，样本数={len(df_bt)}")

    m = evaluate_backtest(df_bt)
    m["method"] = "VAR"

    spent = timing.get("VAR.rolling_backtest", 0.0)
    m = attach_timing(m, spent, n_samples=len(df_bt), prefix="offline_")

    df_metrics = pd.DataFrame([m])[
        ["method", "MAE", "RMSE", "sMAPE", "n",
         "offline_time_sec", "offline_time_per_sample_ms"]
    ]
    summary_csv = OUTPUT_METRICS_DIR / f"layer1_var_summary_chiller_h{HORIZON}.csv"
    df_metrics.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    print("\n=== Layer1 VAR 离线回测汇总 ===")
    print(df_metrics)
    print(f"[汇总] 已保存：{summary_csv}")

    out_png = OUTPUT_FIGURES_DIR / f"layer1_var_chiller_h{HORIZON}.png"
    plot_backtest(
        df_bt,
        title=f"{TARGET_COL} - VAR 多变量离线回测 (h={HORIZON})",
        out_path=out_png,
    )


if __name__ == "__main__":
    run_layer1_var()
