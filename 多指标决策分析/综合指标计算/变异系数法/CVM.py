import numpy as np


def calculate_coefficient_of_variation(data):
    """计算变异系数 (标准差 / 均值)"""
    mean = np.mean(data)
    std_dev = np.std(data)
    return std_dev / mean


def calculate_weights(data_matrix):
    """根据变异系数法计算权重"""
    cvs = np.apply_along_axis(calculate_coefficient_of_variation, 0, data_matrix)
    cvs = 1 / cvs  # 在这里认为变异系数越高的指标越不可靠，权重应该越小
    return cvs / cvs.sum()


if __name__ == "__main__":
    # 示例：比较三组数据的波动性
    data1 = np.array([50, 60, 55, 70, 65])  # 第一个指标
    data2 = np.array([10, 12, 11, 13, 15])  # 第二个指标
    data3 = np.array([200, 210, 190, 205, 195])  # 第三个指标

    cv1 = calculate_coefficient_of_variation(data1)
    cv2 = calculate_coefficient_of_variation(data2)
    cv3 = calculate_coefficient_of_variation(data3)

    print("数据集1的变异系数:", cv1)
    print("数据集2的变异系数:", cv2)
    print("数据集3的变异系数:", cv3)
    
    # 示例数据：三个指标，每列是一个指标的数据
    data_matrix = np.array([
        [50, 60, 70],
        [55, 65, 75],
        [58, 62, 72],
        [60, 68, 80],
        [53, 66, 78]
    ])
    
    weights = calculate_weights(data_matrix)
    print("各指标权重:", weights)
