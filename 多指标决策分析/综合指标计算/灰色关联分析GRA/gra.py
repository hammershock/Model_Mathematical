# 系统分析：在复杂系统中，当变量之间的关系不明确或数据量有限时，GRA可以用来分析不同变量之间的关联程度。
# 预测与决策：在经济预测、市场分析等领域，GRA可以帮助识别关键影响因素，为决策提供支持。
# 质量评估：在产品质量控制中，GRA可以用来比较不同产品或不同生产批次的质量差异。
# 环境评价：评估不同政策对环境影响的程度，例如空气质量、水资源管理等。
# 能源分析：在能源领域，GRA可以用于分析不同能源政策、技术或项目之间的关联和影响程度。

import numpy as np


class GreyRelationalAnalysis:
    def __init__(self, reference_series, comparison_series):
        """
        在样本量不足时，也同样适用的方法
        reference_series: 参考数列，即标准或优化目标
        comparison_series: 比较数列，即待分析的数列
        """
        self.reference_series = np.array(reference_series)
        self.comparison_series = np.array(comparison_series)
    
    def normalize(self, series):
        """数据标准化"""
        min_val = np.min(series)
        max_val = np.max(series)
        return (series - min_val) / (max_val - min_val)
    
    def calculate_correlation(self):
        """计算关联度"""
        # 数据标准化
        norm_ref = self.normalize(self.reference_series)
        norm_comp = np.array([self.normalize(series) for series in self.comparison_series])
        
        # 计算差异序列
        diff_series = np.abs(norm_comp - norm_ref)
        
        # 计算关联系数
        rho = 0.5  # 分辨系数
        correlation_coefficients = (np.min(diff_series) + rho * np.max(diff_series)) / (
                    diff_series + rho * np.max(diff_series))
        
        # 计算关联度
        grey_relation = np.mean(correlation_coefficients, axis=1)
        return grey_relation / grey_relation.sum()


if __name__ == "__main__":
    # 示例：评价三种产品的综合性能
    # 参考数列（最佳性能）
    reference_series = [100, 99, 95]
    # 比较数列（三种产品的实际性能）
    comparison_series = [
        [90, 80, 70],
        [85, 95, 80],
        [70, 60, 90]
    ]
    
    gra = GreyRelationalAnalysis(reference_series, comparison_series)
    results = gra.calculate_correlation()
    print("灰色关联度:", results)
