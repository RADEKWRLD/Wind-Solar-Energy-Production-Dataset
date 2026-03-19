"""
通用评估函数模块

职责：
- 回归指标：MSE、RMSE、MAE、R²
- 分类指标：Accuracy、Precision、Recall、F1-Score、ROC-AUC
- 模型对比表生成：多模型结果汇总为 DataFrame
"""
#这里写评估函数的实现代码，方便后续调用和维护
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

#给回归模型用的评估函数
def regression_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"MSE": mse, "RMSE": rmse, "MAE": mae, "R²": r2}

#给分类模型用的评估函数
def classification_metrics(y_true, y_pred, y_prob=None):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    metrics = {"Accuracy": accuracy, "Precision": precision, "Recall": recall, "F1-Score": f1}
    
    if y_prob is not None:
        try:
            roc_auc = roc_auc_score(y_true, y_prob, multi_class='ovr')
            metrics["ROC-AUC"] = roc_auc
        except Exception as e:
            print(f"计算 ROC-AUC 失败: {e}")
    return metrics