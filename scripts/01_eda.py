"""
探索性数据分析（EDA）

输出目录：outputs/eda/

可视化清单：
- Production 分布直方图（按 Source 分组）
- 各季节/月份/小时的发电量箱线图
- 特征相关性热力图
- 时间趋势折线图（按日/月聚合）
- Wind vs Solar 发电量的日内模式对比
"""
import sys
import os
#之前env，删了一次，忘了加回来了，导致运行报错，补上
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import DataLoader
from src.feature_engineering import FeatureEngineer
from src.visualization import setup_plot_style, save_fig
import seaborn as sns
import matplotlib.pyplot as plt

def main():
    #设置统一图表风格
    setup_plot_style()

    #数据加载与清洗
    data_loader = DataLoader(file_path=os.path.join("datasets", "Energy Production Dataset.csv"))
    df = data_loader.load_and_clean()

    #EDA 用原始数据画图，不做标准化，只加 sin/cos 特征供热力图用
    feature_engineer = FeatureEngineer(data=df)
    df = feature_engineer.date_encoding()

    #基本信息
    print(df.info())
    print(df.describe())

    #图1: Production 分布直方图（按 Source 分组）
    fig, ax = plt.subplots()
    for source in df["Source"].unique():
        subset = df[df["Source"] == source]
        ax.hist(subset["Production"], bins=50, alpha=0.6, label=source)
    ax.set_xlabel("Production (MWh)")
    ax.set_ylabel("频次")
    ax.set_title("发电量分布（按能源类型）")
    ax.legend()
    save_fig(fig, "production_distribution.png", "eda")

    #图2: 各季节/月份/小时的发电量箱线图
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    sns.boxplot(x="Season", y="Production", data=df, ax=axes[0])
    axes[0].set_title("各季节发电量")
    sns.boxplot(x="Month", y="Production", data=df, ax=axes[1])
    axes[1].set_title("各月份发电量")
    sns.boxplot(x="Start_Hour", y="Production", data=df, ax=axes[2])
    axes[2].set_title("各小时发电量")
    fig.tight_layout()
    save_fig(fig, "boxplots.png", "eda")

    #图3: 特征相关性热力图
    fig, ax = plt.subplots(figsize=(12, 8))
    num_cols = df.select_dtypes(include=["number"]).columns
    corr = df[num_cols].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title("特征相关性热力图")
    save_fig(fig, "correlation_heatmap.png", "eda")

    #图4: 时间趋势折线图（按月聚合）
    fig, ax = plt.subplots(figsize=(14, 6))
    monthly = df.groupby([df["Date"].dt.to_period("M"), "Source"])["Production"].mean().reset_index()
    monthly["Date"] = monthly["Date"].dt.to_timestamp()
    for source in monthly["Source"].unique():
        subset = monthly[monthly["Source"] == source]
        ax.plot(subset["Date"], subset["Production"], label=source)
    ax.set_xlabel("日期")
    ax.set_ylabel("平均发电量 (MWh)")
    ax.set_title("月均发电量趋势")
    ax.legend()
    save_fig(fig, "monthly_trend.png", "eda")

    #图5: Wind vs Solar 日内模式对比
    fig, ax = plt.subplots()
    hourly = df.groupby(["Start_Hour", "Source"])["Production"].mean().reset_index()
    for source in hourly["Source"].unique():
        subset = hourly[hourly["Source"] == source]
        ax.plot(subset["Start_Hour"], subset["Production"], marker="o", label=source)
    ax.set_xlabel("小时")
    ax.set_ylabel("平均发电量 (MWh)")
    ax.set_title("Wind vs Solar 日内发电模式")
    ax.legend()
    save_fig(fig, "hourly_pattern.png", "eda")

if __name__ == "__main__":
    main()
