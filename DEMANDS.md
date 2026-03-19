# 风能与太阳能发电数据集 — 机器学习分析需求文档

## 数据集概况

| 项目 | 详情 |
|------|------|
| 来源 | [Kaggle - Wind & Solar Energy Production Dataset](https://www.kaggle.com/datasets/ahmeduzaki/wind-and-solar-energy-production-dataset) |
| 地区 | 法国 |
| 时间跨度 | 2020年1月 ~ 2025年11月 |
| 记录数 | 51,864 条（每小时一条） |
| 列数 | 9 列 |

**列说明**：

| 列名 | 类型 | 说明 |
|------|------|------|
| Date | 日期 | 日期（MM/DD/YYYY） |
| Start_Hour | 整数 | 小时起始（0-23） |
| End_Hour | 整数 | 小时结束 |
| Source | 分类 | 能源类型：Wind / Solar / Mixed |
| Day_of_Year | 整数 | 一年中的第几天（1-366） |
| Day_Name | 分类 | 星期几 |
| Month_Name | 分类 | 月份名 |
| Season | 分类 | 季节：Spring / Summer / Fall / Winter |
| Production | 整数 | 发电量（MWh） |

**关键统计**：
- Wind 占 82%（42,484条），Solar 占 18%（9,378条），Mixed 仅 2 条
- Production 范围：58 ~ 23,446 MWh/小时
- 四季分布基本均匀

---

## 一、数据预处理（Data Preprocessing）

### 1.1 数据清洗
- 删除 Source = "Mixed" 的记录（仅 2 条，无统计意义）
- 检查并处理缺失值（该数据集已标注为完整，仍需验证）

### 1.2 特征工程
- **日期解析**：将 Date 转为 datetime，提取 Year、Month（数值）、Week 等
- **周期性编码**：对 Start_Hour 和 Day_of_Year 做 sin/cos 变换，保留周期性信息
  ```
  hour_sin = sin(2π × Start_Hour / 24)
  hour_cos = cos(2π × Start_Hour / 24)
  day_sin  = sin(2π × Day_of_Year / 365)
  day_cos  = cos(2π × Day_of_Year / 365)
  ```
- **分类编码**：Season、Day_Name 等使用 Label Encoding 或 One-Hot Encoding
- **标准化**：对数值特征进行 StandardScaler / MinMaxScaler 归一化

### 1.3 探索性数据分析（EDA）
- Production 分布直方图（按 Source 分组）
- 各季节/月份/小时的发电量箱线图
- 特征相关性热力图
- 时间趋势折线图（按日/月聚合）
- Wind vs Solar 发电量的日内模式对比

---

## 二、核心任务

### 任务 1：回归 — 发电量预测（Regression）

> **目标**：根据时间特征和能源类型，预测每小时发电量 Production

**输入特征**：
- Start_Hour（sin/cos 编码）
- Day_of_Year（sin/cos 编码）
- Season（编码）
- Source（编码）
- Month（数值）
- Day_Name（编码）

**候选模型**（由简至繁）：

| 模型 | 角色 | 说明 |
|------|------|------|
| Linear Regression | 基线 | 最简单的线性模型，用于设定性能下限 |
| Ridge / Lasso | 正则化对比 | 观察正则化对过拟合的影响 |
| Decision Tree Regressor | 非线性基线 | 捕捉非线性关系 |
| Random Forest Regressor | 集成方法 | Bagging 减少方差 |
| XGBoost / LightGBM | 集成方法 | Boosting 提升精度 |
| SVR | 核方法 | 小数据量时对比（可抽样） |

**评估指标**：MSE、RMSE、MAE、R²

**可视化输出**：
- 模型性能对比柱状图
- 预测值 vs 真实值散点图（最优模型）
- 特征重要性排名图（树模型）

---

### 任务 2：分类 — 能源类型识别（Classification）

> **目标**：根据发电量和时间特征，判断能源来源是 Wind 还是 Solar

**输入特征**：
- Production
- Start_Hour（sin/cos 编码）
- Day_of_Year（sin/cos 编码）
- Season（编码）
- Month（数值）

**类别不均衡处理**（Wind 82% vs Solar 18%）：
- 使用 `class_weight='balanced'` 参数
- 或使用 SMOTE 过采样

**候选模型**：

| 模型 | 说明 |
|------|------|
| Logistic Regression | 线性基线 |
| KNN | 基于距离的分类 |
| Decision Tree | 可解释性强 |
| Random Forest | 集成方法 |
| SVM | 核方法，高维有效 |
| XGBoost | Boosting 集成 |

**评估指标**：Accuracy、Precision、Recall、F1-Score、ROC-AUC

**可视化输出**：
- 模型性能对比表
- 混淆矩阵热力图（最优模型）
- ROC 曲线（多模型叠加对比）
- 决策边界可视化（2D 降维后，可选）

---

### 任务 3：聚类 — 发电模式分析（Clustering）

> **目标**：在不使用 Source 标签的情况下，发现数据中的自然分组模式

**输入特征**：
- Production（标准化）
- Start_Hour（sin/cos 编码）
- Day_of_Year（sin/cos 编码）
- Season（编码）

**候选模型**：

| 模型 | 说明 |
|------|------|
| K-Means | 用肘部法则（Elbow Method）和轮廓系数确定最佳 K |
| DBSCAN | 发现任意形状的簇，自动检测异常值 |
| Agglomerative Clustering | 层次聚类，可绘制树状图 |

**评估指标**：
- 轮廓系数（Silhouette Score）
- Calinski-Harabasz Index

**可视化输出**：
- 肘部法则图（K-Means）
- PCA 降维后的聚类散点图
- 聚类结果 vs 真实 Source 标签的对比分析（交叉表 / Sankey 图）
- 层次聚类树状图（Dendrogram）

---

### 任务 4：时间序列预测（Time Series Forecasting）

> **目标**：基于历史发电趋势，预测未来的日发电总量

**数据准备**：
- 将每小时数据按日聚合为日发电总量
- 分别对 Wind 和 Solar 建模（两条独立时间序列）
- 训练集：2020 ~ 2024 | 测试集：2025

**候选模型**：

| 模型 | 说明 |
|------|------|
| ARIMA / SARIMA | 经典时间序列模型，SARIMA 捕捉季节性 |
| Prophet | Facebook 开源，自动处理趋势 + 季节性 + 节假日 |
| LSTM | 深度学习方法（可选加分项） |

**评估指标**：MSE、RMSE、MAPE

**可视化输出**：
- 时间序列分解图（趋势 + 季节性 + 残差）
- 预测曲线 vs 实际曲线
- 残差分析图
- 预测置信区间

---

## 三、分析报告结构（4000 字+）

| 章节 | 内容要点 |
|------|----------|
| 1. 引言 | 可再生能源背景、数据集介绍、研究目标与意义 |
| 2. 数据预处理 | 清洗步骤、特征工程方法、EDA 关键发现 |
| 3. 回归分析 | 模型选择理由、训练过程、结果对比与分析 |
| 4. 分类分析 | 不均衡处理策略、模型对比、分类效果讨论 |
| 5. 聚类分析 | 聚类方法选择、最佳簇数确定、模式解读 |
| 6. 时间序列预测 | 建模流程、预测效果、趋势讨论 |
| 7. 综合讨论 | 各任务核心发现汇总、模型适用性比较、局限性与改进方向 |
| 8. 结论 | 核心贡献、未来工作展望 |

---

## 四、交付物清单

| 文件 | 说明 |
|------|------|
| `DEMANDS.md` | 需求文档（本文件） |
| `analysis.ipynb` | Jupyter Notebook — 完整代码 + 可视化 + Markdown 说明 |
| `report.md` / `report.pdf` | 分析报告（不少于 4000 字） |
| `outputs/` | 所有图表输出目录 |

---

## 五、技术栈

```
pandas, numpy           — 数据处理
matplotlib, seaborn     — 可视化
scikit-learn            — 回归、分类、聚类、评估
xgboost / lightgbm      — 梯度提升模型
statsmodels / prophet    — 时间序列分析
```
