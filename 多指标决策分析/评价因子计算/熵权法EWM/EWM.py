# -*- coding:utf-8 -*-
# File: EWM.py
# Environment: PyCharm
# Author: Wenjie Xu
# Email: wenjie_xu2000@outlook.com
# Time : 2023/3/26
import numpy as np
from prettytable import PrettyTable, SINGLE_BORDER


class EntropyWeightMethod:
    """Entropy weight method, 熵权法
    一种客观赋权方法，用于确定各个指标在综合评价中的权重
    信息熵越小的指标，其所携带的有效信息越多，因此应赋予更高的权重
    如果某个指标的所有值都很接近，那么这个指标的信息熵就很高，反映出它提供的有效信息较少。
    Parameters
    ----------
    data : ndarray of shape (n_samples, n_features)
        指标矩阵

    data_type : ndarray of shape (n_features,)
        指示向量, 指示各列指标数据是正向指标或负向指标, 1表示正向指标,2表示负向指标, 例如[1,1,2,1]

    scale_min : float, optional, default=0.0001
        归一化的区间端点, 即归一化时将数据缩放到(scale_min, scale_max)的范围内, 默认应设置为(0,1)

    scale_max : float, optional, default=0.9999
        归一化的区间端点, 即归一化时将数据缩放到(scale_min, scale_max)的范围内, 默认应设置为(0,1)

    display : bool, optional, default=True
        是否打印指标权重输出表格

    Returns
    ----------
    y_norm : ndarray of shape (n_samples, n_features)
        归一化后的数据
    score : ndarray of shape (n_features, )
        综合加权评分
    weight : ndarray of shape (n_features,)
        各指标权重
    """

    def __init__(self, data, data_type, scale_min=0, scale_max=1, display=True):
        # 检测输入数据是否为numpy数组
        if not isinstance(data, np.ndarray):
            raise TypeError("指标矩阵必须为numpy.ndarray类型")
        # 检测输入数据是否为二维数组
        if len(data.shape) != 2:
            raise ValueError("指标矩阵必须为二维数组")
        self.data = data
        self.data_type = data_type
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.n, self.m = self.data.shape
        self.display = display

    def transform(self):
        # MinMaxNormalize 归一化
        y_norm = np.zeros((self.n, self.m))
        x_min, x_max = np.min(self.data, axis=0), np.max(self.data, axis=0)
        for i in range(self.m):
            if self.data_type[i] == 1:  # 正向指标归一化
                for j in range(self.m):
                    y_norm[:, j] = ((self.scale_max - self.scale_min) * (self.data[:, j] - x_min[j]) / (
                            x_max[j] - x_min[j]) + self.scale_min).flatten()
            elif self.data_type[i] == 2:  # 负向指标归一化
                for j in range(self.m):
                    y_norm[:, j] = ((self.scale_max - self.scale_min) * (x_max[j] - self.data[:, j]) / (
                            x_max[j] - x_min[j]) + self.scale_min).flatten()
        return y_norm

    def fit(self):
        # EWM熵权法
        y_norm = self.transform()
        # 计算第m项指标下第m个样本值占该指标的比重:比重P(i,j)
        P = np.zeros((self.n, self.m))
        for i in range(self.n):
            for j in range(self.m):
                P[i, j] = y_norm[i, j] / np.sum(y_norm[:, j])
        # 第j个指标的熵值e(j)
        e = np.zeros((1, self.m))
        # 其中k = 1/ln(n)
        k = 1 / np.log10(self.n)
        for j in range(self.m):
            e[0, j] = -k * np.sum(P[:, j] * np.log10(P[:, j]))
        # 计算信息熵冗余度
        d = np.ones_like(e) - e
        # 计算各项指标的权重
        weight = (d / np.sum(d)).flatten()
        # 计算该样本的综合加权评分
        score = np.sum(weight * y_norm, axis=1)
        # 输出结果
        if self.display:
            print_tb = PrettyTable()
            print_tb.add_column("index", np.arange(self.m))
            print_tb.add_column("Index weight", weight)
            print_tb.align = "l"
            print_tb.set_style(SINGLE_BORDER)
            print(print_tb)
        return y_norm, score, weight


if __name__ == "__main__":
    x = np.array([[58.080430, 81.312602, 254.500000, 0.371197, 21.117511, 14.688735],
                  [59.235963, 82.930348, 274.400000, 0.387443, 22.768742, 15.837284],
                  [60.444956, 84.622938, 297.080179, 0.404609, 24.650663, 17.146294],
                  [61.685770, 86.360078, 320.129288, 0.422775, 26.563197, 18.476598],
                  [62.936763, 88.111469, 344.174671, 0.442023, 28.558398, 19.864402],
                  [64.176296, 89.846814, 369.843672, 0.462434, 30.688321, 21.345916],
                  [65.382726, 91.535817, 397.763635, 0.484088, 33.005021, 22.957346],
                  [66.534414, 93.148179, 428.561905, 0.507068, 35.560552, 24.734900]])
    # 假设一共有六个指标，所有的指标都是正向指标，越高越好
    # 正向指标的data_type为1，负向指标的data_type为2
    y, s, w = EntropyWeightMethod(data=x, data_type=[1, 1, 1, 1, 1, 1], scale_min=0.0001, scale_max=0.9999, display=True).fit()
    