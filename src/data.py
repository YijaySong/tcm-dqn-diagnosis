import pandas as pd  # 导入Pandas库，用于数据操作
import numpy as np  # 导入NumPy库，用于数值计算

def load_and_preprocess_data(filepath):  # 定义函数加载和预处理数据
    try:
        # Load the data with space as the delimiter 使用空格作为分隔符加载数据
        # data = pd.readc_csv(filepath, delimiter=' ', encoding='utf-8') # 从文件中读取数据，使用空格作为分隔符，并设置编码为utf-8
        data = pd.read_csv(filepath, sep='\s+', encoding='utf-8') # 从文件中读取数据，使用空格作为分隔符，并设置编码为utf-8
        print(f"Successfully loaded the file with encoding: utf-8 and delimiter: ' '") # 打印成功加载文件的信息
    except Exception as e: # 捕获加载文件时的异常
        print(f"Failed to load the file: {str(e)}") # 打印错误信息
        return None, None # 如果加载失败，返回None

    # Define expected columns 定义预期的列名称
    expected_columns = ['编号', '病种分类', '病证名', '中医症状', '中医证型', '状态要素'] # 定义数据集的预期列
    actual_columns = list(data.columns) # 获取数据集中实际的列名称
    print(f"Actual columns in the file: {actual_columns}") # 打印文件中的实际列名称

    # Check for missing columns 检查是否有缺失的列
    missing_columns = [col for col in expected_columns if col not in actual_columns] # 找出预期列中缺失的列
    if missing_columns: # 如果存在缺失列
        raise ValueError(f"Missing columns in the dataset: {missing_columns}. Expected columns: {expected_columns}") # 抛出异常提示缺失的列
    # print(data)
    # If columns are correct, continue preprocessing 如果列正确，则继续进行预处理
    features = data.drop(columns=['编号', '病种分类', '病证名','状态要素'])  # Example, modify as needed 示例：删除不需要的列，保留特征列
    print(features)
    labels = data['状态要素'] # 将'中医证型'列作为标签

    return features, labels # 返回特征和标签

def create_datasets(features, labels, test_size=0.2): # 定义函数创建训练集和测试集
    # Create train and test datasets 创建训练集和测试集
    from sklearn.model_selection import train_test_split # 导入函数用于划分数据集
    train_features, test_features, train_labels, test_labels = train_test_split(  # 划分数据集，按比例将数据划分为训练集和测试集
        features, labels, test_size=test_size, random_state=42)
    
    # Convert to Dataset objects (if using PyTorch or similar) 如果使用PyTorch或类似库，则将数据转换为Dataset对象
    train_dataset = Dataset(train_features, train_labels) # 创建训练集对象
    test_dataset = Dataset(test_features, test_labels) # 创建测试集对象
    
    return train_dataset, test_dataset # 返回训练集和测试集

class Dataset: # 定义Dataset类
    def __init__(self, features, labels): # 初始化函数
        self.features = features # 初始化特征
        self.labels = labels # 初始化标签
    
    def __len__(self): # 定义返回数据集长度的方法
        return len(self.features) # 返回特征的数量
    
    def __getitem__(self, idx): # 定义获取特定样本的方法
        return self.features.iloc[idx], self.labels.iloc[idx] # 返回指定索引的特征和标签
