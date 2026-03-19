"""
任务3：发电模式聚类分析

输出目录：outputs/clustering/

模型：K-Means / DBSCAN / Agglomerative Clustering
评估：轮廓系数（Silhouette Score）、Calinski-Harabasz Index

可视化清单：
- 肘部法则图（K-Means）
- PCA 降维后的聚类散点图
- 聚类结果 vs 真实 Source 标签的对比分析
- 层次聚类树状图（Dendrogram）
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from scipy.cluster.hierarchy import dendrogram, linkage

from src.data_loader import DataLoader
from src.feature_engineering import FeatureEngineer
from src.visualization import setup_plot_style, save_fig


def main():
    setup_plot_style()

    #数据加载
    loader = DataLoader(file_path=os.path.join("datasets", "Energy Production Dataset.csv"))
    df = loader.load_and_clean()
    fe = FeatureEngineer(data=df)
    df = fe.date_encoding()

    #保留真实标签用于后续对比
    true_labels = df["Source"].copy()

    #聚类输入特征（不使用 Source 标签）
    feature_cols = ["Production", "Hour_Sin", "Hour_Cos", "DayOfYear_Sin", "DayOfYear_Cos", "Month", "Quarter"]

    #标准化
    scaler = StandardScaler()
    X = scaler.fit_transform(df[feature_cols])

    #图1: 肘部法则图（确定最佳 K）
    #轮廓系数用抽样计算，避免全量计算太慢
    inertias = []
    sil_scores = []
    K_range = range(2, 11)

    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X)
        inertias.append(km.inertia_)
        sil_scores.append(silhouette_score(X, km.labels_, sample_size=25000, random_state=42))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(K_range, inertias, "bo-")
    axes[0].set_xlabel("K")
    axes[0].set_ylabel("Inertia")
    axes[0].set_title("肘部法则")
    axes[1].plot(K_range, sil_scores, "ro-")
    axes[1].set_xlabel("K")
    axes[1].set_ylabel("轮廓系数")
    axes[1].set_title("轮廓系数随 K 变化")
    fig.tight_layout()
    save_fig(fig, "elbow_method.png", "clustering")

    #根据轮廓系数选最佳 K
    best_k = list(K_range)[np.argmax(sil_scores)]
    print(f"最佳 K = {best_k}（轮廓系数 = {max(sil_scores):.4f}）")

    #训练三个聚类模型
    #Agglomerative 在 5 万条上很慢，用抽样训练后对全量数据用最近质心分配
    models = {
        "K-Means": KMeans(n_clusters=best_k, random_state=42, n_init=10),
        "DBSCAN": DBSCAN(eps=1.5, min_samples=10),
    }

    results = {}
    all_labels = {}

    for name, model in models.items():
        print(f"训练 {name}...")
        labels = model.fit_predict(X)
        all_labels[name] = labels

    #Agglomerative
    print("训练 Agglomerative...")
    np.random.seed(42)
    agg_sample_idx = np.random.choice(len(X), 25000, replace=False)
    agg_model = AgglomerativeClustering(n_clusters=best_k)
    agg_sample_labels = agg_model.fit_predict(X[agg_sample_idx])

    #用 KMeans 的质心思路：对每个簇算质心，然后全量数据按最近质心分配
    from sklearn.metrics import pairwise_distances_argmin_min
    centroids = np.array([X[agg_sample_idx][agg_sample_labels == c].mean(axis=0) for c in range(best_k)])
    agg_full_labels, _ = pairwise_distances_argmin_min(X, centroids)
    all_labels["Agglomerative"] = agg_full_labels

    for name, labels in all_labels.items():

        n_clusters = len(set(labels) - {-1})
        if n_clusters >= 2:
            sil = silhouette_score(X, labels, sample_size=25000, random_state=42)
            ch = calinski_harabasz_score(X, labels)
            results[name] = {"簇数": n_clusters, "轮廓系数": sil, "CH Index": ch}
        else:
            results[name] = {"簇数": n_clusters, "轮廓系数": None, "CH Index": None}
        print(f"{name}: {results[name]}")

    #PCA 降维到 2D
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    #图2: PCA 降维后的聚类散点图（三个模型 + 真实标签）
    fig, axes = plt.subplots(1, 4, figsize=(24, 5))

    #真实标签
    for label in true_labels.unique():
        mask = true_labels == label
        axes[0].scatter(X_pca[mask, 0], X_pca[mask, 1], alpha=0.3, s=10, label=label)
    axes[0].set_title("真实标签")
    axes[0].legend()

    #三个聚类模型
    for idx, (name, labels) in enumerate(all_labels.items()):
        scatter = axes[idx + 1].scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap="viridis", alpha=0.3, s=10)
        axes[idx + 1].set_title(name)
        plt.colorbar(scatter, ax=axes[idx + 1])

    for ax in axes:
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")

    fig.tight_layout()
    save_fig(fig, "pca_clusters.png", "clustering")

    #图3: 聚类结果 vs 真实 Source 标签的对比（交叉表）
    km_labels = all_labels["K-Means"]
    cross_tab = pd.crosstab(true_labels, km_labels, rownames=["真实标签"], colnames=["K-Means 簇"])
    print("\nK-Means 聚类 vs 真实标签交叉表:")
    print(cross_tab)

    fig, ax = plt.subplots()
    cross_tab.plot.bar(ax=ax)
    ax.set_title("K-Means 聚类 vs 真实 Source 标签")
    ax.set_ylabel("样本数")
    ax.set_xlabel("真实标签")
    ax.legend(title="簇编号")
    save_fig(fig, "cluster_vs_true.png", "clustering")

    #图4: 层次聚类树状图（抽样，避免图太密）
    sample_size = 500
    np.random.seed(42)
    sample_idx = np.random.choice(len(X), sample_size, replace=False)
    X_sample = X[sample_idx]

    Z = linkage(X_sample, method="ward")

    fig, ax = plt.subplots(figsize=(14, 6))
    dendrogram(Z, truncate_mode="lastp", p=30, ax=ax)
    ax.set_title("层次聚类树状图")
    ax.set_xlabel("样本")
    ax.set_ylabel("距离")
    save_fig(fig, "dendrogram.png", "clustering")


if __name__ == "__main__":
    main()
