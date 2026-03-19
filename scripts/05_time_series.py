"""
任务4：时间序列预测

输出目录：outputs/time_series/

数据准备：每小时 → 按日聚合，Wind/Solar 分别建模
训练集：2020~2024 | 测试集：2025
模型：ARIMA / SARIMA / Prophet / LSTM（可选）
评估：MSE、RMSE、MAPE

可视化清单：
- 时间序列分解图（趋势 + 季节性 + 残差）
- 预测曲线 vs 实际曲线
- 残差分析图
- 预测置信区间
"""
