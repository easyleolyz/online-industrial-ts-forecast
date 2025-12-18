# src/online_ts_forecast/traditional/run_layer1_var_online_chiller.py
#预热后进行模型的评估
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from online_ts_forecast.traditional.var_models import var_forecast
from online_ts_forecast.traditional.online_eval import online_eval_multivar_var
from online_ts_forecast.traditional.metrics import evaluate_backtest


# ======= 配置 =======

DATA_PATH = Path("data/private/tem_data_07.csv")
TIME_COL_INDEX = 0

TARGET_COL = "冷机3总有功功率"
COVARIATE_COLS = [
    "冷机3总有功功率",
    "冷机4总有功功率",
]

FREQ_MINUTES = 5
HORIZON = 1
WARMUP_DAYS = 7
EVAL_DAYS = 7
MIN_HISTORY = 500
ONLINE_STRIDE_STEPS = 6
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


def plot_online(df: pd.DataFrame, title: str, out_path: Path) -> None:
    if df.empty:
        print(f"[WARN] 空在线结果，跳过绘图：{title}")
        return

    t = df["obs_time"]
    y = df["y_true"]
    yhat = df["y_pred"]

    plt.figure(figsize=(12, 4))
    plt.plot(t, y, label="真实", linewidth=1.2)
    plt.plot(t, yhat, label="在线预测", linewidth=1.0)

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


def var_forecast_wrapper(history: pd.DataFrame, horizon: int, maxlags: int = 12) -> pd.DataFrame:
    return var_forecast(history, horizon=horizon, maxlags=maxlags)


def run_layer1_var_online():
    df_multi = load_chiller_multivar()

    OUTPUT_METRICS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("\n=== [Layer1 在线] VAR 多变量评估 ===")
    df_online = online_eval_multivar_var(
        df=df_multi,
        target_col=TARGET_COL,
        var_forecast_func=var_forecast_wrapper,
        horizon=HORIZON,
        warmup_days=WARMUP_DAYS,
        eval_days=EVAL_DAYS,
        freq_minutes=FREQ_MINUTES,
        min_history=MIN_HISTORY,
        stride_steps=ONLINE_STRIDE_STEPS,
        maxlags=MAXLAGS,
    )

    out_csv = OUTPUT_METRICS_DIR / f"var_online_chiller_h{HORIZON}_warmup{WARMUP_DAYS}d_eval{EVAL_DAYS}d.csv"
    df_online.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"[CSV] 已保存：{out_csv}，样本数={len(df_online)}")

    m = evaluate_backtest(
        df_online.rename(columns={"obs_time": "time"})
    )
    m["method"] = "VAR_online"
    df_metrics = pd.DataFrame([m])[["method", "MAE", "RMSE", "sMAPE", "n"]]

    summary_csv = OUTPUT_METRICS_DIR / f"layer1_var_online_summary_chiller_h{HORIZON}_warmup{WARMUP_DAYS}d_eval{EVAL_DAYS}d.csv"
    df_metrics.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    print("\n=== Layer1 VAR 在线评估汇总 ===")
    print(df_metrics)
    print(f"[汇总] 已保存：{summary_csv}")

    out_png = OUTPUT_FIGURES_DIR / f"var_online_chiller_h{HORIZON}_warmup{WARMUP_DAYS}d_eval{EVAL_DAYS}d.png"
    plot_online(
        df_online,
        title=f"{TARGET_COL} - VAR 多变量在线评估 (h={HORIZON}, warmup={WARMUP_DAYS}d, eval={EVAL_DAYS}d)",
        out_path=out_png,
    )


if __name__ == "__main__":
    run_layer1_var_online()
