# 项目文件架构

```
Wind&Solar-Energy-Production-Dataset/
│
├── .gitignore
├── DEMANDS.md                          # 需求文档
├── ARCH.md                             # 架构文档（本文件）
├── requirements.txt                    # Python 依赖
├── main.py                             # 统一入口，一键运行全部任务
│
├── datasets/                           # 原始数据（gitignore）
│   └── wind_solar_production.csv
│
├── src/                                # 公共模块
│   ├── __init__.py
│   ├── data_loader.py                  # 数据加载与清洗
│   ├── feature_engineering.py          # 特征工程
│   ├── evaluation.py                   # 通用评估函数
│   └── visualization.py               # 通用可视化函数
│
├── scripts/                            # 各任务脚本（按执行顺序编号）
│   ├── 01_eda.py                       # 探索性数据分析
│   ├── 02_regression.py                # 任务1：发电量回归预测
│   ├── 03_classification.py            # 任务2：能源类型分类
│   ├── 04_clustering.py                # 任务3：发电模式聚类
│   └── 05_time_series.py              # 任务4：时间序列预测
│
├── outputs/                            # 图表输出（gitignore）
│   ├── eda/
│   ├── regression/
│   ├── classification/
│   ├── clustering/
│   └── time_series/
│
├── report/                             # 分析报告
│   └── report.md
│
└── logs/                               # 训练日志（gitignore）
```

---

## src/ 公共模块

| 模块 | 职责 | 被哪些脚本使用 |
|------|------|----------------|
| `data_loader.py` | `load_data()` — 读取 CSV、删除 Mixed 记录、验证缺失值、日期解析并提取 Year/Month/Week | 所有脚本 |
| `feature_engineering.py` | `encode_cyclic()` sin/cos 周期编码、`encode_categorical()` 标签/One-Hot 编码、`scale_features()` 标准化 | 02-05 |
| `evaluation.py` | `regression_metrics()` MSE/RMSE/MAE/R²、`classification_metrics()` Accuracy/F1/ROC-AUC、`compare_models()` 模型对比表 | 02-05 |
| `visualization.py` | `setup_style()` 统一图表风格、`save_fig()` 自动保存到对应 outputs/ 子目录、各类通用绘图函数 | 所有脚本 |

---

## scripts/ 任务脚本

| 脚本 | 对应 DEMANDS 任务 | 输出目录 |
|------|-------------------|----------|
| `01_eda.py` | §1.3 EDA — Production 分布图、季节/月份/小时箱线图、相关性热力图、时间趋势图、Wind vs Solar 日内对比 | `outputs/eda/` |
| `02_regression.py` | §任务1 — Linear/Ridge/Lasso/DecisionTree/RandomForest/XGBoost 回归对比、特征重要性、预测 vs 真实散点图 | `outputs/regression/` |
| `03_classification.py` | §任务2 — LogisticRegression/KNN/DecisionTree/RandomForest/SVM/XGBoost 分类、class_weight/SMOTE 不均衡处理、ROC 曲线、混淆矩阵 | `outputs/classification/` |
| `04_clustering.py` | §任务3 — K-Means/DBSCAN/Agglomerative 聚类、肘部法则、PCA 降维散点图、聚类 vs 真实标签对比、树状图 | `outputs/clustering/` |
| `05_time_series.py` | §任务4 — 日聚合、Wind/Solar 分别建模、ARIMA/Prophet/LSTM、时序分解图、预测曲线、残差分析 | `outputs/time_series/` |

---

## 运行方式

```bash
# 运行单个任务
python scripts/01_eda.py

# 一键运行全部
python main.py
```
