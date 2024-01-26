# -*- coding:utf-8 -*-
# File: AHP.py
# Environment: PyCharm
# Author: Wenjie Xu
# Email: wenjie_xu2000@outlook.com
# Time : 2023/3/26

import numpy as np
from prettytable import PrettyTable, SINGLE_BORDER


class AHP_Method:
    """Analytic Hierarchy Process, 层次分析法
    多标准决策分析方法，用于容易通过比较给出的重要性等级得出所有评估指标的重要性权重
    
    Parameters
    ----------
    data : ndarray of shape (n_features, n_features)
        判断矩阵, 重要性标度有1,3,5,7,9和倒数等, 倒用分数表示，而不是小数。支持30阶内的判断矩阵

    display : bool, optional, default=True
        是否打印计算结果输出表格

    Returns
    ----------
    max_variable : float
        判断矩阵最大的特征值

    max_vector : ndarray of shape (n_features,)
        判断矩阵最大特征值对应的特征向量

    weight_vector : ndarray of shape (n_features,)
        归一化后的权重向量
    """

    def __init__(self, data, display=True):
        # 检测输入数据是否为numpy数组
        if not isinstance(data, np.ndarray):
            raise TypeError("判断矩阵必须为numpy.ndarray类型")
        # 检测输入数据是否满足正互反矩阵的要求
        if not self.is_positive_reciprocal_matrix(data):
            raise ValueError("判断矩阵必须为正互反矩阵, 请检查输入数据，不要输入零、小数或者负数")
        # 检测输入数据是否为30阶内的判断矩阵
        if data.shape[0] > 30:
            raise ValueError("判断矩阵阶数不得超过30阶")
        self.data = data
        self.size = data.shape[0]  # 指标数量
        value_RI = [0, 0, 0.52, 0.89, 1.12, 1.26, 1.36, 1.41, 1.46, 1.49, 1.52, 1.54, 1.56, 1.58, 1.59, 1.5934, 1.6064,
                    1.6133, 1.6207, 1.6292, 1.6358, 1.6403, 1.6462, 1.6497, 1.6556, 1.6587, 1.6631, 1.6670, 1.6693,
                    1.6724]  # 随机一致性RI表，30阶，可扩展
        self.RI = value_RI[self.size - 1]  # 随机一致性RI值
        self.max_var = None  # 初始化判断矩阵最大的特征值
        self.display = display  # 是否打印计算结果输出表格
        self.print_tb_ci = PrettyTable()  # 初始化检验输出表格

    # 检测判断矩阵是否为正互反矩阵
    @staticmethod
    def is_positive_reciprocal_matrix(test_data):
        n, m = test_data.shape  # 获取矩阵的形状和阶数
        # 检查是否是方阵，所有元素都大于零，对角线上的元素都等于一，任意两个非对角线元素是否满足a_ij * a_ji = 1
        return n == m and np.all(test_data > 0) and np.all(np.diag(test_data) == 1) and np.isclose(
            (test_data.T * test_data == 1).all(), True)

    # 计算归一化的权重向量
    def cal_eigen(self):
        # 计算判断矩阵的特征值与特征向量
        eigen_value, eigen_vector = np.linalg.eig(self.data)
        # 计算矩阵的最大特征值与其对应的特征向量
        max_variable = np.max(eigen_value)
        index = np.argmax(eigen_value)
        max_variable = round(max_variable.real, 4)
        max_vector = eigen_vector[:, index].real.round(4)
        self.max_var = max_variable
        # 计算归一化的权重向量W
        weight_vector = max_vector / sum(max_vector)
        weight_vector = weight_vector.round(4)
        # 输出结果
        if self.display:
            print_tb = PrettyTable()
            print_tb.field_names = ["AHP results", "Value"]
            print_tb.add_rows([["Largest eigenvalue of the judgment matrix", max_variable],
                               ["Eigenvector corresponding to the maximum eigenvalue", max_vector],
                               ["Normalized weight vector", weight_vector], ])
            print_tb.align = "l"  # 设置右对齐
            print_tb.set_style(SINGLE_BORDER)  # 设置表格样式
            print(print_tb)
        return max_variable, max_vector, weight_vector

    # 检验判断矩阵的一致性
    def check_consistency(self):
        # 计算判断矩阵的CI值
        CI = (self.max_var - self.size) / (self.size - 1)
        CI = round(CI, 4)
        # 输出结果
        self.print_tb_ci = PrettyTable()
        self.print_tb_ci.field_names = ["Inspection", "Value"]
        self.print_tb_ci.add_rows([["CI value of the judgment matrix", CI],
                                   ["RI value of the judgment matrix", self.RI], ])
        self.print_tb_ci.align = "l"
        self.print_tb_ci.set_style(SINGLE_BORDER)
        if self.size == 2:
            self.print_tb_ci.add_row(
                ["Results", "Judgment matrix only has two variables, and there is no consistency problem"])
            return print(self.print_tb_ci)
        else:
            # 计算CR值
            CR = CI / self.RI
            CR = round(CR, 4)
            if CR < 0.10:
                self.print_tb_ci.add_row(
                    ["Results", "Judgment matrix CR value {}, passed consistency check".format(CR)])
                return print(self.print_tb_ci)
            else:
                self.print_tb_ci.add_row(
                    ["Results", "Judgment matrix CR value {}, failed consistency check".format(CR)])
                return print(self.print_tb_ci)


if __name__ == "__main__":
    # =========================判断矩阵的实际意义=========================
    # aij 表示第i个元素相对于第j个元素的重要性
    # 1: 同等重要
    # 3: 略为重要
    # 5: 明显重要
    # 以此类推
    # 这个矩阵是正互反的，即aij = 1 / aji
    judgement_data = np.array([[1, 1 / 3, 1 / 5, 1 / 3, 1 / 5, 1 / 5],
                               [3, 1, 1 / 3, 1 / 3, 1 / 5, 1 / 5],
                               [5, 3, 1, 1, 1 / 3, 1 / 3],
                               [3, 3, 1, 1, 1, 1],
                               [5, 5, 3, 1, 1, 1],
                               [5, 5, 3, 1, 1, 1]])
    
    model = AHP_Method(judgement_data, display=True)
    max_var, max_v, weight = model.cal_eigen()
    model.check_consistency()


# ### 代码输出结果分析
#
# 1. **最大特征值和特征向量**:
#    - 最大特征值为6.306，这是判断矩阵的一个重要特征值。
#    - 对应的特征向量是[-0.0916, -0.1424, -0.3192, -0.415, -0.5904, -0.5904]，它表示了各个元素相对权重的一个未归一化估计。
#
# 2. **归一化权重向量**:
#    - 归一化后的权重向量是[0.0426, 0.0663, 0.1485, 0.1931, 0.2747, 0.2747]，这表示了各元素在决策中的相对重要性。
#    - 权重最高的是最后两个元素，意味着在决策中它们的优先级最高。
#
# 3. **一致性检验**:
#    - CI值（一致性指标）为0.0612，表示判断矩阵的一致性水平。
#    - RI值（随机一致性指标）为1.26，是预先定义的，与判断矩阵的阶数有关。
#    - CR值（一致性比率）为0.0486，由CI/RI计算得出。因为CR小于0.10，所以判断矩阵通过了一致性检验，说明决策者的判断具有合理的一致性。
#
# 综上，这个AHP模型的判断矩阵不仅提供了决策元素间相对重要性的量化评估，而且其一致性是可接受的，这说明决策过程是可靠的。
