"""
特征工程模块

职责：
- 周期性编码：Start_Hour、Day_of_Year → sin/cos 变换
- 分类编码：Season、Day_Name → Label Encoding / One-Hot Encoding
- 标准化：数值特征 → StandardScaler / MinMaxScaler
"""
import numpy as np
import pandas as pd
import os
from src.utils.singleton import Singleton

#特征工程类
@Singleton
class FeatureEngineer:
    def __init__(self,data):
        self.data = data
    
    #提取时间特征 + 周期性编码
    def date_encoding(self):
        #时间特征
        self.data["Quarter"] = self.data["Date"].dt.quarter
        self.data["IsWeekend"] = self.data["Day_Name"].isin(["Saturday", "Sunday"]).astype(int)

        #周期性编码
        #小时（0–23）和年终日期（1–366）是循环的 ：第 23 小时接近第 0 小时，12 月 31 日接近 1 月 1 日。
        #标准整数编码在这些值之间制造了人为的“距离”。正弦/余弦变换保持循环关系，显著提升了回归和神经网络模型的性能。
        self.data["Hour_Sin"] = np.sin(2 * np.pi * self.data["Start_Hour"] / 24)
        self.data["Hour_Cos"] = np.cos(2 * np.pi * self.data["Start_Hour"] / 24)
        self.data["DayOfYear_Sin"] = np.sin(2 * np.pi * self.data["Day_of_Year"] / 366)
        self.data["DayOfYear_Cos"] = np.cos(2 * np.pi * self.data["Day_of_Year"] / 366)

        #有序分类（让图表按正确顺序排列）
        month_order = ["January","February","March","April","May","June",
                       "July","August","September","October","November","December"]
        day_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        season_order = ["Winter","Spring","Summer","Fall"]

        self.data["Month_Name"] = pd.Categorical(self.data["Month_Name"], categories=month_order, ordered=True)
        self.data["Day_Name"] = pd.Categorical(self.data["Day_Name"], categories=day_order, ordered=True)
        self.data["Season"] = pd.Categorical(self.data["Season"], categories=season_order, ordered=True)

        return self.data
    
    #分类编码，这里我为了对不同模型，写两种分类编码
    #适合树模型：Label Encoding
    #适合线性模型：One-Hot Encoding
    #这里用 method 做传参，args:"label"/"onehot"
    #需要做标签分类的根据 dataset 有四种，我直接写个通用的解法了
    def categorical_encoding(self,method="label"):
        cols = ["Source", "Day_Name", "Month_Name", "Season"]
        if method == "label":
            from sklearn.preprocessing import LabelEncoder
            for col in cols:
                le = LabelEncoder()
                self.data[col] = le.fit_transform(self.data[col])
        elif method == "onehot":
            self.data = pd.get_dummies(self.data, columns=cols)
        return self.data
    
    #数据标准化
    def standardize(self):
        from sklearn.preprocessing import StandardScaler
        num_cols = self.data.select_dtypes(include=[np.number]).columns
        scaler = StandardScaler()
        self.data[num_cols] = scaler.fit_transform(self.data[num_cols])
        return self.data