import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt
import numpy as np

# 对于特征之间有相关性的情况，可以将其分解为不相关的一些主成分
Data = load_breast_cancer()
data = Data.data  # 获得数据矩阵
target = Data.target  # 获得分类目标

print(data.shape)
print(Data.feature_names)

pca = PCA(n_components=5, random_state=123)  # 初始化主成分分析模型
pca.fit(data)  # 拟合数据
print(pd.DataFrame(pca.explained_variance_ratio_))  # 查看各个主成分的方差贡献率
pca_result = pca.transform(data)[:, :1]  # 投影数据, 仅仅保留第一个主成分

# ======================================SVM======================================
X_train, X_test, y_train, y_test = train_test_split(pca_result, target, test_size=0.3, random_state=123)

svm = SVC()
svm.fit(X_train, y_train)

y_pred = svm.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# ========================Plot boundary========================

# 在一维上绘制数据点
plt.scatter(X_train, y_train, color='blue', label='Training data')
plt.scatter(X_test, y_test, color='green', label='Testing data')

# 绘制SVM决策边界
ax = plt.gca()
xlim = ax.get_xlim()
xx = np.linspace(xlim[0], xlim[1], 30)
yy = svm.predict(xx.reshape(-1, 1))
plt.plot(xx, yy, color='red', label='Decision boundary')

plt.xlabel('First Principal Component')
plt.ylabel('Class')
plt.title('SVM on First Principal Component')
plt.legend()
plt.show()

