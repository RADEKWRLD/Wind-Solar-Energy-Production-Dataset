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
│   └── Energy Production Dataset.csv   # 51,864 条逐小时发电记录
│
├── src/                                # 公共模块
│   ├── __init__.py
│   ├── data_loader.py                  # 数据加载与清洗（Singleton）
│   ├── feature_engineering.py          # 特征工程（Singleton）
│   ├── evaluation.py                   # 通用评估函数
│   ├── visualization.py                # 通用可视化函数
│   └── utils/
│       ├── __init__.py
│       └── singleton.py                # Singleton 装饰器
│
├── scripts/                            # 各任务脚本（按执行顺序编号）
│   ├── 01_eda.py                       # 探索性数据分析
│   ├── 02_regression.py                # 任务1：发电量回归预测
│   ├── 03_classification.py            # 任务2：能源类型分类
│   ├── 04_clustering.py                # 任务3：发电模式聚类
│   └── 05_time_series.py              # 任务4：时间序列预测
│
├── outputs/                            # 图表输出（gitignore）
│   ├── eda/                            # 5 张图：分布、箱线图、热力图、月度趋势、日内模式
│   ├── regression/                     # 3 张图：模型对比、预测散点、特征重要性
│   ├── classification/                 # 3 张图：模型对比、混淆矩阵、ROC 曲线
│   ├── clustering/                     # 4 张图：肘部法则、PCA 散点、聚类对比、树状图
│   └── time_series/                    # 8 张图：Wind/Solar 各 4 张（分解、预测、残差、置信区间）
│
├── report/                             # 分析报告
│   └── report.md                       # 完整分析报告（≥4000 字）
│
└── logs/                               # 训练日志（gitignore）
    ├── 01_eda_{timestamp}.log
    ├── 02_regression_{timestamp}.log
    ├── 03_classification_{timestamp}.log
    ├── 04_clustering_{timestamp}.log
    └── 05_time_series_{timestamp}.log
```

---

## src/ 公共模块

| 模块 | 职责 | 被哪些脚本使用 |
|------|------|----------------|
| `data_loader.py` | `DataLoader` 类（Singleton）— 读取 CSV、删除 Mixed 记录、验证缺失值、日期解析并提取 Year/Month/Week | 所有脚本 |
| `feature_engineering.py` | `FeatureEngineer` 类（Singleton）— `date_encoding()` sin/cos 周期编码 + Quarter/IsWeekend、`categorical_encoding()` 标签/One-Hot 编码、`standardize()` StandardScaler 标准化 | 02-05 |
| `evaluation.py` | `regression_metrics()` MSE/RMSE/MAE/R²、`classification_metrics()` Accuracy/Precision/Recall/F1/ROC-AUC、模型对比表输出 | 02-05 |
| `visualization.py` | `setup_plot_style()` 统一图表风格（含中文字体）、`save_fig()` 自动保存到对应 outputs/ 子目录 | 所有脚本 |
| `utils/singleton.py` | Singleton 装饰器，防止 DataLoader/FeatureEngineer 重复实例化 | data_loader, feature_engineering |

---

## scripts/ 任务脚本

| 脚本 | 对应 DEMANDS 任务 | 输出目录 |
|------|-------------------|----------|
| `01_eda.py` | §1.3 EDA — Production 分布图、季节/月份/小时箱线图、相关性热力图、时间趋势图、Wind vs Solar 日内对比 | `outputs/eda/` |
| `02_regression.py` | §任务1 — Linear/Ridge/Lasso/DecisionTree/RandomForest/XGBoost 回归对比、特征重要性、预测 vs 真实散点图 | `outputs/regression/` |
| `03_classification.py` | §任务2 — LogisticRegression/KNN/DecisionTree/RandomForest/SVM/XGBoost 分类、`class_weight='balanced'` 不均衡处理、ROC 曲线、混淆矩阵 | `outputs/classification/` |
| `04_clustering.py` | §任务3 — K-Means/DBSCAN/Agglomerative 聚类、肘部法则、PCA 降维散点图、聚类 vs 真实标签对比、树状图 | `outputs/clustering/` |
| `05_time_series.py` | §任务4 — 日聚合、Wind/Solar 分别建模、SARIMA/Prophet 时序预测、时序分解图、预测曲线、残差分析、置信区间 | `outputs/time_series/` |

---

## main.py 运行机制

`main.py` 通过子进程（`subprocess`）按顺序执行 5 个脚本，每个脚本的标准输出和错误输出重定向至 `logs/` 目录下带时间戳的日志文件。

---

## 运行方式

```bash
# 运行单个任务
python scripts/01_eda.py

# 一键运行全部
python main.py
```

---

## 技术栈

```
pandas, numpy           — 数据处理
matplotlib, seaborn     — 可视化
scikit-learn            — 回归、分类、聚类、评估
xgboost                 — 梯度提升模型
statsmodels             — SARIMA 时间序列
prophet                 — Facebook Prophet 时间序列预测
```
