"""
任务4：时间序列预测

输出目录：outputs/time_series/

数据准备：每小时 → 按日聚合，Wind/Solar 分别建模
训练集：2020~2024 | 测试集：2025
模型：SARIMA / Prophet / LSTM / Transformer
评估：MSE、RMSE、MAPE

可视化清单：
- 时间序列分解图（趋势 + 季节性 + 残差）
- 预测曲线 vs 实际曲线
- 残差分析图
- 预测置信区间
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader as TorchDataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet

from src.data_loader import DataLoader
from src.evaluation import regression_metrics
from src.visualization import setup_plot_style, save_fig

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
SEQ_LEN = 30  # 用过去30天预测下一天
EPOCHS = 80
BATCH_SIZE = 32
LR = 1e-3


def mape(y_true, y_pred):
    """计算 MAPE"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class TransformerModel(nn.Module):
    def __init__(self, input_size=1, d_model=64, nhead=4, num_layers=2, dropout=0.2):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_enc = nn.Parameter(torch.randn(1, SEQ_LEN, d_model) * 0.01)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=128,
            dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.input_proj(x) + self.pos_enc
        x = self.encoder(x)
        return self.fc(x[:, -1, :])


def create_sequences(data, seq_len):
    """将一维序列转为 (X, y) 滑窗样本"""
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i + seq_len])
        y.append(data[i + seq_len])
    return np.array(X), np.array(y)


def train_torch_model(model, train_series, test_series, model_name):
    """通用 PyTorch 模型训练 + 预测流程"""
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_series.values.reshape(-1, 1)).flatten()
    full_scaled = scaler.transform(
        np.concatenate([train_series.values, test_series.values]).reshape(-1, 1)
    ).flatten()

    X_train, y_train = create_sequences(train_scaled, SEQ_LEN)
    X_train = torch.FloatTensor(X_train).unsqueeze(-1).to(DEVICE)
    y_train = torch.FloatTensor(y_train).unsqueeze(-1).to(DEVICE)

    dataset = TensorDataset(X_train, y_train)
    loader = TorchDataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(EPOCHS):
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

    # 滚动预测测试集
    model.eval()
    preds = []
    # 用训练集最后 SEQ_LEN 天作为初始窗口
    window = list(train_scaled[-SEQ_LEN:])
    with torch.no_grad():
        for _ in range(len(test_series)):
            x = torch.FloatTensor([window[-SEQ_LEN:]]).unsqueeze(-1).to(DEVICE)
            pred = model(x).cpu().item()
            preds.append(pred)
            window.append(pred)

    preds = scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
    return preds


def forecast_source(df_source, source_name):
    """对单个能源类型做时间序列预测"""
    print(f"\n{'='*50}")
    print(f"正在处理: {source_name}")
    print(f"{'='*50}")

    #按日聚合
    daily = df_source.groupby("Date")["Production"].sum().reset_index()
    daily = daily.set_index("Date").sort_index()
    daily = daily.asfreq("D", fill_value=0)

    #划分训练集/测试集（2020-2024 训练，2025 测试）
    train = daily[daily.index.year <= 2024]
    test = daily[daily.index.year >= 2025]
    print(f"训练集: {train.index.min()} ~ {train.index.max()} ({len(train)} 天)")
    print(f"测试集: {test.index.min()} ~ {test.index.max()} ({len(test)} 天)")

    #图1: 时间序列分解图
    decomposition = seasonal_decompose(train["Production"], model="additive", period=365)
    fig, axes = plt.subplots(4, 1, figsize=(14, 10))
    decomposition.observed.plot(ax=axes[0])
    axes[0].set_title("原始数据")
    decomposition.trend.plot(ax=axes[1])
    axes[1].set_title("趋势")
    decomposition.seasonal.plot(ax=axes[2])
    axes[2].set_title("季节性")
    decomposition.resid.plot(ax=axes[3])
    axes[3].set_title("残差")
    fig.suptitle(f"{source_name} 时间序列分解", fontsize=16)
    fig.tight_layout()
    save_fig(fig, f"decomposition_{source_name.lower()}.png", "time_series")
    results = {}

    #SARIMA
    print(f"\n训练 SARIMA ({source_name})...")
    try:
        sarima = SARIMAX(
            train["Production"],
            order=(1, 1, 1),
            seasonal_order=(1, 1, 1, 7),
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        sarima_fit = sarima.fit(disp=False)
        sarima_pred = sarima_fit.forecast(steps=len(test))
        sarima_pred.index = test.index

        metrics = regression_metrics(test["Production"], sarima_pred)
        metrics["MAPE"] = mape(test["Production"], sarima_pred)
        results["SARIMA"] = {"pred": sarima_pred, "metrics": metrics}
        print(f"SARIMA: {metrics}")
    except Exception as e:
        print(f"SARIMA 失败: {e}")

    #Prophet
    print(f"\n训练 Prophet ({source_name})...")
    try:
        prophet_train = train.reset_index().rename(columns={"Date": "ds", "Production": "y"})

        prophet = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
        prophet.fit(prophet_train)

        future = prophet.make_future_dataframe(periods=len(test))
        forecast = prophet.predict(future)

        prophet_pred = forecast.set_index("ds").loc[test.index, "yhat"]

        metrics = regression_metrics(test["Production"], prophet_pred)
        metrics["MAPE"] = mape(test["Production"], prophet_pred)
        results["Prophet"] = {"pred": prophet_pred, "metrics": metrics}
        print(f"Prophet: {metrics}")
    except Exception as e:
        print(f"Prophet 失败: {e}")

    #LSTM
    print(f"\n训练 LSTM ({source_name})...")
    try:
        lstm_model = LSTMModel(input_size=1, hidden_size=64, num_layers=2)
        lstm_preds = train_torch_model(lstm_model, train["Production"], test["Production"], "LSTM")
        lstm_pred_series = pd.Series(lstm_preds, index=test.index)

        metrics = regression_metrics(test["Production"], lstm_pred_series)
        metrics["MAPE"] = mape(test["Production"], lstm_preds)
        results["LSTM"] = {"pred": lstm_pred_series, "metrics": metrics}
        print(f"LSTM: {metrics}")
    except Exception as e:
        print(f"LSTM 失败: {e}")

    #Transformer
    print(f"\n训练 Transformer ({source_name})...")
    try:
        transformer_model = TransformerModel(input_size=1, d_model=64, nhead=4, num_layers=2)
        transformer_preds = train_torch_model(transformer_model, train["Production"], test["Production"], "Transformer")
        transformer_pred_series = pd.Series(transformer_preds, index=test.index)

        metrics = regression_metrics(test["Production"], transformer_pred_series)
        metrics["MAPE"] = mape(test["Production"], transformer_preds)
        results["Transformer"] = {"pred": transformer_pred_series, "metrics": metrics}
        print(f"Transformer: {metrics}")
    except Exception as e:
        print(f"Transformer 失败: {e}")

    if not results:
        print(f"{source_name}: 所有模型都失败了")
        return

    #图2: 预测曲线 vs 实际曲线
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(test.index, test["Production"], label="实际值", color="black", linewidth=1.5)
    colors = ["blue", "red", "green", "orange", "purple"]
    for idx, (name, data) in enumerate(results.items()):
        ax.plot(test.index, data["pred"], label=f"{name}", color=colors[idx], alpha=0.7)
    ax.set_xlabel("日期")
    ax.set_ylabel("日发电量 (MWh)")
    ax.set_title(f"{source_name} 预测 vs 实际")
    ax.legend()
    save_fig(fig, f"forecast_{source_name.lower()}.png", "time_series")

    #图3: 残差分析图（最优模型）
    best_name = min(results, key=lambda k: results[k]["metrics"]["RMSE"])
    best_pred = results[best_name]["pred"]
    residuals = test["Production"] - best_pred

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(test.index, residuals)
    axes[0].axhline(y=0, color="r", linestyle="--")
    axes[0].set_title(f"残差时序图（{best_name}）")
    axes[0].set_xlabel("日期")
    axes[0].set_ylabel("残差")

    axes[1].hist(residuals, bins=50, edgecolor="black")
    axes[1].set_title(f"残差分布（{best_name}）")
    axes[1].set_xlabel("残差")
    axes[1].set_ylabel("频次")

    fig.suptitle(f"{source_name} 残差分析", fontsize=16)
    fig.tight_layout()
    save_fig(fig, f"residuals_{source_name.lower()}.png", "time_series")

    #图4: 预测置信区间（Prophet）
    if "Prophet" in results:
        fig, ax = plt.subplots(figsize=(14, 6))
        forecast_test = forecast.set_index("ds").loc[test.index]
        ax.plot(test.index, test["Production"], label="实际值", color="black")
        ax.plot(test.index, forecast_test["yhat"], label="Prophet 预测", color="blue")
        ax.fill_between(test.index, forecast_test["yhat_lower"], forecast_test["yhat_upper"],
                        alpha=0.2, color="blue", label="95% 置信区间")
        ax.set_xlabel("日期")
        ax.set_ylabel("日发电量 (MWh)")
        ax.set_title(f"{source_name} Prophet 预测置信区间")
        ax.legend()
        save_fig(fig, f"confidence_interval_{source_name.lower()}.png", "time_series")

    #打印对比表
    metrics_df = pd.DataFrame({name: data["metrics"] for name, data in results.items()}).T
    print(f"\n{source_name} 模型对比:")
    print(metrics_df)


def main():
    setup_plot_style()

    #数据加载
    loader = DataLoader(file_path=os.path.join("datasets", "Energy Production Dataset.csv"))
    df = loader.load_and_clean()

    #Wind 和 Solar 分别建模
    for source in ["Wind", "Solar"]:
        df_source = df[df["Source"] == source].copy()
        forecast_source(df_source, source)


if __name__ == "__main__":
    main()
