"""
通用可视化模块

职责：
- 统一图表风格设置
- save_fig()：自动保存图表到对应 outputs/ 子目录
- 通用绘图函数：箱线图、热力图、混淆矩阵、ROC 曲线等
"""

#这里写可视化函数的实现代码，方便后续调用和维护
import os
import matplotlib.pyplot as plt
import seaborn as sns

#统一图表风格设置
def setup_plot_style():
    sns.set_style("whitegrid")
    plt.rcParams["figure.figsize"] = (10, 6)
    plt.rcParams["axes.titlesize"] = 16
    plt.rcParams["axes.labelsize"] = 14
    plt.rcParams["xtick.labelsize"] = 12
    plt.rcParams["ytick.labelsize"] = 12
    plt.rcParams["legend.fontsize"] = 12

#自动保存图表到对应 outputs/ 子目录
def save_fig(fig, filename, subdir="general"):
    output_dir = os.path.join("outputs", subdir)
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, filename)
    fig.savefig(save_path, bbox_inches='tight')
    print(f"图表已保存到 {save_path}")