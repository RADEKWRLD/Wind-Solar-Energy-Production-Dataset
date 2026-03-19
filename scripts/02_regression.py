"""
任务1：发电量回归预测

输出目录：outputs/regression/

模型：Linear Regression / Ridge / Lasso / Decision Tree / Random Forest / XGBoost
评估：MSE、RMSE、MAE、R²

可视化清单：
- 模型性能对比柱状图
- 预测值 vs 真实值散点图（最优模型）
- 特征重要性排名图（树模型）
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from src.data_loader import DataLoader
from src.feature_engineering import FeatureEngineer
from src.evaluation import regression_metrics
from src.visualization import setup_plot_style, save_fig


def main():
    setup_plot_style()

    #数据加载
    loader = DataLoader(file_path=os.path.join("datasets", "Energy Production Dataset.csv"))
    df = loader.load_and_clean()

    fe = FeatureEngineer(data=df)
    df = fe.date_encoding()

    y = df["Production"]

    #共用的数值特征,可以不做处理
    common_cols = ["Hour_Sin", "Hour_Cos", "DayOfYear_Sin", "DayOfYear_Cos", "Month", "Quarter", "IsWeekend"]

    #其实都用 label 都可以，但是我为了数据的严谨性就用了两套编码，线性模型用 onehot，树模型用 label，这样更符合实际项目中的做法
    #Label 编码数据（给树模型用）
    df_label = df.copy()
    fe_label = FeatureEngineer(data=df_label)
    df_label = fe_label.categorical_encoding(method="label")
    label_cols = common_cols + ["Source", "Season", "Day_Name", "Month_Name"]
    X_label = df_label[label_cols]

    #One-Hot 编码数据（给线性模型用）
    df_onehot = df.copy()
    fe_onehot = FeatureEngineer(data=df_onehot)
    df_onehot = fe_onehot.categorical_encoding(method="onehot")
    onehot_cols = [c for c in df_onehot.columns if c not in ["Production", "Date", "Start_Hour", "End_Hour", "Day_of_Year"]]
    X_onehot = df_onehot[onehot_cols]

    #划分训练集/测试集
    X_label_train, X_label_test, y_train, y_test = train_test_split(
        X_label, y, test_size=0.2, random_state=42
    )
    X_onehot_train, X_onehot_test, _, _ = train_test_split(
        X_onehot, y, test_size=0.2, random_state=42
    )

    #线性模型用 onehot，树模型用 label
    models = {
        "Linear Regression": (LinearRegression(), "onehot"),
        "Ridge": (Ridge(alpha=1.0), "onehot"),
        "Lasso": (Lasso(alpha=1.0), "onehot"),
        "Decision Tree": (DecisionTreeRegressor(random_state=42), "label"),
        "Random Forest": (RandomForestRegressor(n_estimators=100, random_state=42), "label"),
        "XGBoost": (XGBRegressor(n_estimators=100, random_state=42), "label"),
    }

    #输出结果字典
    results = {}
    predictions = {}
    trained_models = {}

    for name, (model, encoding) in models.items():
        if encoding == "onehot":
            Xtr, Xte = X_onehot_train, X_onehot_test
        else:
            Xtr, Xte = X_label_train, X_label_test

        model.fit(Xtr, y_train)
        y_pred = model.predict(Xte)
        metrics = regression_metrics(y_test, y_pred)
        results[name] = metrics
        predictions[name] = y_pred
        trained_models[name] = model
        print(f"{name}: {metrics}")

    #模型对比表
    results_df = pd.DataFrame(results).T
    print("\n模型对比:")
    print(results_df)

    #图1: 模型性能对比柱状图
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    results_df["R²"].sort_values().plot.barh(ax=axes[0])
    axes[0].set_title("R² 对比")
    axes[0].set_xlabel("R²")
    results_df["RMSE"].sort_values().plot.barh(ax=axes[1])
    axes[1].set_title("RMSE 对比")
    axes[1].set_xlabel("RMSE")
    fig.tight_layout()
    save_fig(fig, "model_comparison.png", "regression")

    #图2: 预测值 vs 真实值散点图（R² 最高的模型）
    best_model_name = results_df["R²"].idxmax()
    best_pred = predictions[best_model_name]
    fig, ax = plt.subplots()
    ax.scatter(y_test, best_pred, alpha=0.3, s=10)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
    ax.set_xlabel("真实值 (MWh)")
    ax.set_ylabel("预测值 (MWh)")
    ax.set_title(f"预测 vs 真实（{best_model_name}）")
    save_fig(fig, "pred_vs_true.png", "regression")

    #图3: 特征重要性排名图
    rf_model = trained_models["Random Forest"]
    importances = rf_model.feature_importances_
    feat_imp = pd.Series(importances, index=label_cols).sort_values()

    fig, ax = plt.subplots()
    feat_imp.plot.barh(ax=ax)
    ax.set_title("特征重要性（Random Forest）")
    ax.set_xlabel("重要性")
    save_fig(fig, "feature_importance.png", "regression")


if __name__ == "__main__":
    main()