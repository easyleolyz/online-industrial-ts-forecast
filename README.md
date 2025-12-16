# 在线工业时序预测 / Online Industrial Time Series Forecasting

本仓库用于整理与复现本人在数据中心冷机 / 服务器资源等场景下的**在线时序预测**工作，
并扩展到传统时间序列算法学习、金融时序预测（如 AUM、风控、市场波动）等。

## 开发与运行环境

- 开发环境：Windows + WSL2（Ubuntu）+ VS Code (Remote - WSL)
- 运行环境：
  - 本地 WSL2：快速调试、脚本运行
  - Google Colab：训练、绘图、传统算法学习与复现（记录环境与版本）

代码目录放在 Windows 的 `D:\ProjectTime` 下，
在 WSL2 中对应路径为 `/mnt/d/ProjectTime`。

## 数据隐私

工业冷机 / 服务器资源等数据为私有数据，不随仓库分发。
仓库仅包含数据格式与字段说明（见 `data/README.md`）。

## 目录结构概览

- `src/online_ts_forecast/`：主库代码
  - `traditional/`：传统时间序列算法实现与封装（ARIMA/ETS/VAR 等）
  - `models/`：机器学习/深度学习模型（LGBM-Quantile, TFT, DeepAR 等）
  - `online/`：在线预测循环、Conformal 校准、候选融合与路由器
  - `industrial/`：工业冷机场景（你的 LGBM 在线仿真脚本在这里落地）
  - `finance/`：未来扩展到 AUM / 风控 / 市场波动预测
- `notebooks/`：学习与实验 Notebook（在 Colab 或本地运行）
  - `traditional/`：系统性学习传统算法，保留实验过程与图像
- `experiments/`：每个实验一份配置（yml）+ 一份结果（csv），便于复盘与比较
- `api/`：FastAPI 在线预测服务（后续接入）
- `docker/`：容器化部署配置

后续会逐步加入：
- 传统算法全家桶的 Notebook 与代码实现
- 工业冷机在线预测的可复现实验与评估
- 金融时序（如 AUM）预测的 Demo 场景
