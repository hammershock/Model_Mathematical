"""
优劣解距离法，被应用于多指标的最优决策
"""
import numpy as np
import pandas as pd


def entropy_weight(features):
    # 熵权法
    features = np.array(features)
    proportion = features / features.sum(axis=0)  # normalized, 归一化, 得到每个样本的比例
    entropy = np.nansum(-proportion * np.log(proportion) / np.log(len(features)), axis=0)  # calculate entropy
    weight = (1 - entropy) / (1 - entropy).sum()
    return weight  # calculation weight coefficient


def topsis(data, weight=None):
    data = data / np.linalg.norm(data, axis=0)  # normalize along the dimension of samples, for each feature
    # best and worst solution
    Z = pd.DataFrame([data.min(), data.max()], index=['Negative Ideal Solution', 'Positive Ideal Solution'])
    weight = entropy_weight(data) if weight is None else np.array(weight)  # importance weight for each dimension of feature
    Result = data.copy()
    Result['Distance to Positive Ideal Solution'] = np.sqrt(((data - Z.loc['Positive Ideal Solution']) ** 2 * weight).sum(axis=1))
    Result['Distance to Negative Ideal Solution'] = np.sqrt(((data - Z.loc['Negative Ideal Solution']) ** 2 * weight).sum(axis=1))
    
    # composite score index
    Result['score'] = Result['Distance to Negative Ideal Solution'] / (Result['Distance to Negative Ideal Solution'] + Result['Distance to Positive Ideal Solution'])
    Result['rank'] = Result.rank(ascending=False)['score']
    
    return Result, Z, weight


# 示例数据：三个方案（A, B, C），每个方案有四个评价指标
data = pd.DataFrame({
    'plan A': [0.8, 0.9, 0.7, 0.6],
    'plan B': [7.7, 0.8, 0.8, 0.7],
    'plan C': [7.9, 0.7, 0.6, 0.8]
}).T

# 使用TOPSIS方法进行评价
result, ideal_solutions, weights = topsis(data)

# 输出结果
print(result)
