"""
特征工程模块

职责：
- 周期性编码：Start_Hour、Day_of_Year → sin/cos 变换
- 分类编码：Season、Day_Name → Label Encoding / One-Hot Encoding
- 标准化：数值特征 → StandardScaler / MinMaxScaler
"""
import numpy as np
import pandas as pd

#特征工程类
class FeatureEngineer:
    def __init__(self,data):
        self.data = data
    
    #周期性编码
    def date_encoding(self):
        self.data["Hour_Sin"] = np.sin(2 * np.pi * self.data["Start_Hour"] / 24)
        self.data["Hour_Cos"] = np.cos(2 * np.pi * self.data["Start_Hour"] / 24)
        self.data["Day_Sin"] = np.sin(2 * np.pi * self.data["Day_of_Year"] / 365)
        self.data["Day_Cos"] = np.cos(2 * np.pi * self.data["Day_of_Year"] / 365)
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