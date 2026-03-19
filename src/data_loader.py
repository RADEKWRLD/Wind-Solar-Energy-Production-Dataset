"""
数据加载与清洗模块

职责：
- 读取 CSV 数据集
- 删除 Source="Mixed" 的记录
- 检查并处理缺失值
- 日期解析：Date → datetime，提取 Year、Month（数值）、Week
"""
