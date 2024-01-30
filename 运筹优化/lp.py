import pulp

# ==================整数线性规划法==================
# 创建一个线性规划问题实例，目标是最大化利润
model = pulp.LpProblem("Maximize Profit", pulp.LpMaximize)

# 定义决策变量
x_A = pulp.LpVariable("x_A", lowBound=0, cat='Integer')  # 产品A的生产数量
x_B = pulp.LpVariable("x_B", lowBound=0, cat='Integer')  # 产品B的生产数量

# 定义目标函数（总利润）
profit = 20 * x_A + 30 * x_B
model += profit

# 定义约束条件
model += x_A + 2 * x_B <= 9  # 生产时间约束
model += x_A <= 5            # 产品A的生产上限
model += x_B <= 3            # 产品B的生产上限

# 求解问题
model.solve()

# 输出结果
print(f"Production of Product A: {x_A.varValue}")
print(f"Production of Product B: {x_B.varValue}")
print(f"Total Profit: {pulp.value(model.objective)}")
