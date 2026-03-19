"""
数据加载与清洗模块

职责：
- 读取 CSV 数据集
- 删除 Source="Mixed" 的记录
- 检查并处理缺失值
- 日期解析：Date → datetime，提取 Year、Month（数值）、Week
"""
import pandas as pd
import os 
import numpy as np
from utils.singleton import Singleton

#数据清洗类
@Singleton
class DataLoader:
    def __init__(self,file_path):
        self.file_path = file_path
        self.data = None
        
    #加载数据   
    def load_data(self):
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"文件 {self.file_path} 不存在,前往官网下载数据集")
        try:
            df = pd.read_csv(self.file_path)
            #传值
            self.data = df
            return df
        except Exception as e:
            raise IOError(f"读取文件 {self.file_path} 失败: {e}")

    #数据清洗
    def clean_data(self):
        if self.data is None:
            raise ValueError("数据未加载，请先调用 load_data() 方法")
        
        #删除 Source="Mixed" 的记录
        self.data = self.data[self.data['Source'] != 'Mixed']
        
        #检查并处理缺失值
        self.data.replace('', np.nan, inplace=True)
        self.data.dropna(inplace=True)
        
        #日期解析
        self.data['Date'] = pd.to_datetime(self.data['Date'], errors='coerce')
        self.data['Year'] = self.data['Date'].dt.year
        self.data['Month'] = self.data['Date'].dt.month
        self.data['Week'] = self.data['Date'].dt.isocalendar().week
        
        return self.data
    
    def load_and_clean(self):
        self.load_data()
        return self.clean_data()