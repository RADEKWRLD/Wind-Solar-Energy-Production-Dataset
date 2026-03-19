"""
任务2：能源类型分类

输出目录：outputs/classification/

模型：Logistic Regression / KNN / Decision Tree / Random Forest / SVM / XGBoost
不均衡处理：class_weight='balanced' / SMOTE
评估：Accuracy、Precision、Recall、F1-Score、ROC-AUC

可视化清单：
- 模型性能对比表
- 混淆矩阵热力图（最优模型）
- ROC 曲线（多模型叠加对比）
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, roc_curve, auc

from src.data_loader import DataLoader
from src.feature_engineering import FeatureEngineer
from src.evaluation import classification_metrics
from src.visualization import setup_plot_style, save_fig


def main():
    setup_plot_style()

    #数据加载
    loader = DataLoader(file_path=os.path.join("datasets", "Energy Production Dataset.csv"))
    df = loader.load_and_clean()
    fe = FeatureEngineer(data=df)
    df = fe.date_encoding()

    #目标变量：Source（Wind=1, Solar=0）
    le = LabelEncoder()
    y = le.fit_transform(df["Source"])

    #输入特征（不包含 Source）
    common_cols = ["Production", "Hour_Sin", "Hour_Cos", "DayOfYear_Sin", "DayOfYear_Cos", "Month", "Quarter", "IsWeekend"]

    #Label 编码数据（给树模型用）
    df_label = df.copy()
    fe_label = FeatureEngineer(data=df_label)
    df_label = fe_label.categorical_encoding(method="label")
    label_cols = common_cols + ["Season", "Day_Name", "Month_Name"]
    X_label = df_label[label_cols]

    #One-Hot 编码数据（给线性模型用）
    df_onehot = df.copy()
    fe_onehot = FeatureEngineer(data=df_onehot)
    df_onehot = fe_onehot.categorical_encoding(method="onehot")
    onehot_cols = [c for c in df_onehot.columns if c not in ["Production", "Date", "Start_Hour", "End_Hour", "Day_of_Year", "Source"]]
    onehot_cols = ["Production"] + onehot_cols
    X_onehot = df_onehot[onehot_cols]

    #划分训练集/测试集
    X_label_train, X_label_test, y_train, y_test = train_test_split(
        X_label, y, test_size=0.2, random_state=42, stratify=y
    )
    X_onehot_train, X_onehot_test, _, _ = train_test_split(
        X_onehot, y, test_size=0.2, random_state=42, stratify=y
    )

    #线性模型用 onehot + class_weight='balanced'，树模型用 label
    models = {
        "Logistic Regression": (LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42), "onehot"),
        "KNN": (KNeighborsClassifier(n_neighbors=5), "onehot"),
        "Decision Tree": (DecisionTreeClassifier(class_weight="balanced", random_state=42), "label"),
        "Random Forest": (RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42), "label"),
        "SVM": (SVC(class_weight="balanced", probability=True, random_state=42), "onehot"),
        "XGBoost": (XGBClassifier(n_estimators=100, scale_pos_weight=len(y[y==0])/len(y[y==1]), random_state=42), "label"),
    }

    results = {}
    predictions = {}
    probabilities = {}
    trained_models = {}

    for name, (model, encoding) in models.items():
        if encoding == "onehot":
            Xtr, Xte = X_onehot_train, X_onehot_test
        else:
            Xtr, Xte = X_label_train, X_label_test

        model.fit(Xtr, y_train)
        y_pred = model.predict(Xte)
        y_prob = model.predict_proba(Xte)[:, 1]

        metrics = classification_metrics(y_test, y_pred, y_prob)
        results[name] = metrics
        predictions[name] = y_pred
        probabilities[name] = y_prob
        trained_models[name] = model
        print(f"{name}: {metrics}")

    #模型对比表
    results_df = pd.DataFrame(results).T
    print("\n模型对比:")
    print(results_df)

    #图1: 模型性能对比柱状图
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    results_df["F1-Score"].sort_values().plot.barh(ax=axes[0])
    axes[0].set_title("F1-Score 对比")
    axes[0].set_xlabel("F1-Score")
    results_df["ROC-AUC"].sort_values().plot.barh(ax=axes[1])
    axes[1].set_title("ROC-AUC 对比")
    axes[1].set_xlabel("ROC-AUC")
    fig.tight_layout()
    save_fig(fig, "model_comparison.png", "classification")

    #图2: 混淆矩阵热力图（F1 最高的模型）
    best_model_name = results_df["F1-Score"].idxmax()
    best_pred = predictions[best_model_name]
    cm = confusion_matrix(y_test, best_pred)

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=le.classes_, yticklabels=le.classes_, ax=ax)
    ax.set_xlabel("预测标签")
    ax.set_ylabel("真实标签")
    ax.set_title(f"混淆矩阵（{best_model_name}）")
    save_fig(fig, "confusion_matrix.png", "classification")

    #图3: ROC 曲线（多模型叠加对比）
    fig, ax = plt.subplots()
    for name, y_prob in probabilities.items():
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", label="随机分类")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC 曲线对比")
    ax.legend(loc="lower right", fontsize=9)
    save_fig(fig, "roc_curves.png", "classification")

if __name__ == "__main__":
    main()
