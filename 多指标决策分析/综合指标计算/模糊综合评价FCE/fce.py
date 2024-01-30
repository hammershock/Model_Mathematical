import numpy as np

# 环境评估：评估一个区域的环境质量，考虑因素可能包括空气质量、水质、噪音水平等。由于这些因素可能具有主观性和模糊性，FCE可以提供一个量化的评价方法。
# 社会经济分析：比如评估城市的宜居度，涉及的因素可能包括经济发展、交通便利、教育资源等。FCE可以帮助分析这些复杂且模糊的因素。
# 项目评价：在项目管理中，使用FCE来综合评估项目的风险、成本效益、进度符合度等多个指标。
# 产品评价：评估产品的综合性能，包括质量、成本、用户满意度等。


class FuzzyComprehensiveEvaluation:
    def __init__(self, factors_weight, evaluation_matrix):
        """
        ————源于模糊数学
        
        factors_weight: 评价因子的权重数组
        evaluation_matrix: 评价矩阵，每一行对应一个对象，每一列对应一个评价因子的评分
        """
        self.factors_weight = np.array(factors_weight)
        self.evaluation_matrix = np.array(evaluation_matrix)
        assert self.factors_weight.shape[0] == self.evaluation_matrix.shape[1], "权重与评价矩阵的维度不匹配"

    def evaluate(self):
        """
        进行模糊综合评价
        """
        # 归一化处理
        normalized_weights = self.factors_weight / np.sum(self.factors_weight)
        # 通过模糊综合运算得到综合评价结果
        comprehensive_evaluation = np.dot(self.evaluation_matrix, normalized_weights)  # 一种模糊矩阵的合成运算
        return comprehensive_evaluation


if __name__ == "__main__":
    # =====================单层次模糊综合评价=====================
    # 示例1：评价三个项目的综合表现
    # 评价因子权重（例如：质量、成本、效率）
    factors_weight = [0.3, 0.4, 0.3]
    
    # 三个项目对应的评价矩阵
    evaluation_matrix = [
        [0.8, 0.6, 0.7],  # 项目1的评分 / 项目1的隶属度得分
        [0.9, 0.8, 0.6],  # 项目2的评分 / 项目2的隶属度得分
        [0.7, 0.9, 0.8]   # 项目3的评分 / 项目3的隶属度得分
    ]
    # 隶属度函数需要经过构造得出
    # 一级模糊综合评价与根据隶属度加权是一样的
    
    evaluator = FuzzyComprehensiveEvaluation(factors_weight, evaluation_matrix)
    results = evaluator.evaluate()
    print("综合评价结果:", results)
    
    # 示例2：等级评价
    # 评价因子权重
    factors_weight = [0.3, 0.4, 0.2, 0.1]
    
    # 三个项目对应的评价矩阵
    evaluation_matrix = [
        [0.7, 0.1, 0.1, 0.1],  # 领导1的评分
        [0.3, 0.3, 0.2, 0.2],  # 领导2的评分
        [0.3, 0.3, 0.1, 0.3]  # 领导3的评分
    ]

    evaluator = FuzzyComprehensiveEvaluation(factors_weight, evaluation_matrix)
    results = evaluator.evaluate()
    print("综合评价结果:", results)  # 三个评价等级的隶属度
    # =====================多层次模糊综合评价=====================
    ...
