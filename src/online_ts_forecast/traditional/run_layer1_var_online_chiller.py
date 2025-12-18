# src/online_ts_forecast/traditional/run_layer1_var_online_chiller.py

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import rcParams
from tqdm.auto import tqdm

from online_ts_forecast.traditional.metrics import evaluate_backtest
from online_ts_forecast.traditional.var_models import var_forecast
from online_ts_forecast.utils.timing import time_block, attach_timing

# ---- 中文字体 ----
rcParams["font.sans-serif"] = ["SimHei"]
rcParams["axes.unicode_minus"] = False

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


def _define_online_window(
    times: pd.DatetimeIndex,
    warmup_days: int,
    eval_days: int,
) -> Tuple[pd.Timestamp, pd.Timestamp]:
    t0 = times.min()
    warmup_end = t0 + pd.Timedelta(days=warmup_days)
    eval_end = warmup_end + pd.Timedelta(days=eval_days)
    return warmup_end, eval_end


def online_eval_multivar_var_with_tqdm(
    df: pd.DataFrame,
    target_col: str,
    method_name: str,
    horizon: int,
    warmup_days: int,
    eval_days: int,
    freq_minutes: int,
    min_history: int,
    stride_steps: int,
    maxlags: int,
) -> pd.DataFrame:
    """
    VAR 在线评估（本文件局部版本），带 tqdm 进度条。
    """
    df = df.dropna()
    df = df.sort_index()
    times = df.index

    warmup_end, eval_end = _define_online_window(times, warmup_days, eval_days)
    freq = pd.Timedelta(minutes=freq_minutes)

    eval_mask = (times > warmup_end) & (times <= eval_end)
    eval_times = times[eval_mask]

    rows = []

    if len(eval_times) == 0:
        print("[WARN] 在线评估窗口为空。")
        return pd.DataFrame(columns=["obs_time", "y_true", "y_pred"])

    iterator = enumerate(eval_times)
    iterator = tqdm(iterator, total=len(eval_times), desc=f"{method_name} online")

    for i, obs_time in iterator:
        if i % stride_steps != 0:
            continue

        issue_time = obs_time - horizon * freq
        history = df[df.index <= issue_time]
        if len(history) < min_history:
            continue

        df_fcst = var_forecast(history, horizon=horizon, maxlags=maxlags)
        if df_fcst is None or len(df_fcst) < horizon:
            continue

        try:
            y_true = float(df.loc[obs_time][target_col])
        except KeyError:
            continue
        y_pred = float(df_fcst[target_col].iloc[horizon - 1])

        rows.append(
            {
                "obs_time": obs_time,
                "y_true": y_true,
                "y_pred": y_pred,
            }
        )

    return pd.DataFrame(rows)


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


def run_layer1_var_online():
    timing_global: Dict[str, float] = {}
    with time_block("load_chiller_multivar", timing_global):
        df_multi = load_chiller_multivar()

    OUTPUT_METRICS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("\n=== [Layer1 在线] VAR 多变量评估 ===")
    timing: Dict[str, float] = {}

    with time_block("VAR.online_eval", timing):
        df_online = online_eval_multivar_var_with_tqdm(
            df=df_multi,
            target_col=TARGET_COL,
            method_name="VAR_online",
            horizon=HORIZON,
            warmup_days=WARMUP_DAYS,
            eval_days=EVAL_DAYS,
            freq_minutes=FREQ_MINUTES,
            min_history=MIN_HISTORY,
            stride_steps=ONLINE_STRIDE_STEPS,
            maxlags=MAXLAGS,
        )

    out_csv = (
        OUTPUT_METRICS_DIR
        / f"var_online_chiller_h{HORIZON}_warmup{WARMUP_DAYS}d_eval{EVAL_DAYS}d.csv"
    )
    df_online.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"[CSV] 已保存：{out_csv}，样本数={len(df_online)}")

    m = evaluate_backtest(df_online.rename(columns={"obs_time": "time"}))
    m["method"] = "VAR_online"

    spent = timing.get("VAR.online_eval", 0.0)
    m = attach_timing(m, spent, n_samples=len(df_online), prefix="online_")

    df_metrics = pd.DataFrame([m])[
        [
            "method",
            "MAE",
            "RMSE",
            "sMAPE",
            "n",
            "online_time_sec",
            "online_time_per_sample_ms",
        ]
    ]
    summary_csv = (
        OUTPUT_METRICS_DIR
        / f"layer1_var_online_summary_chiller_h{HORIZON}_warmup{WARMUP_DAYS}d_eval{EVAL_DAYS}d.csv"
    )
    df_metrics.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    print("\n=== Layer1 VAR 在线评估汇总 ===")
    print(df_metrics)
    print(f"[汇总] 已保存：{summary_csv}")

    out_png = (
        OUTPUT_FIGURES_DIR
        / f"var_online_chiller_h{HORIZON}_warmup{WARMUP_DAYS}d_eval{EVAL_DAYS}d.png"
    )
    plot_online(
        df_online,
        title=(
            f"{TARGET_COL} - VAR 多变量在线评估 "
            f"(h={HORIZON}, warmup={WARMUP_DAYS}d, eval={EVAL_DAYS}d)"
        ),
        out_path=out_png,
    )


if __name__ == "__main__":
    run_layer1_var_online()
