- GitHub Flavored Markdown不支持Latex渲染，请下载后使用VScode或Obsidian等软件配套食用
- 参考来源：[知乎：熵权法确定权重](https://zhuanlan.zhihu.com/p/28067337)

# 基本原理

- 在信息论中，**熵是对不确定性的一种度量**。
- 不确定性越大，熵就越大，包含的信息量越大；不确定性越小，熵就越小，包含的信息量就越小。
- 根据熵的特性，可以通过计算熵值来判断一个事件的随机性及无序程度，也可以用熵值来判断某个指标的离散程度，指标的离散程度越大，该指标对综合评价的影响（权重）越大。比如样本数据在某指标下取值都相等，则该指标对总体评价的影响为0，权值为0.
- 熵权法是一种客观赋权法，因为它仅依赖于数据本身的离散性。

# 熵值法步骤

1. 对$n$个样本，$m$个指标，则$x_{ij}$为第$i$个样本的第$j$个指标的数值$(i=1,\cdots, n; j=1,\cdots,m)$
2. 指标的归一化处理：异质指标同质化
    - 由于各项指标的**计量单位并不统一**，因此在用它们计算综合指标前，先要进行标准化处理，即把指标的绝对值转化为相对值，从而解决各项不同质指标值的同质化问题。
    - 另外，**正向指标和负向指标数值代表的含义不同**（正向指标数值越高越好，负向指标数值越低越好），因此，对于正向负向指标需要采用不同的算法进行数据标准化处理：
    - 为了方便起见，归一化后的数据$x_{ij}^{'}$仍记为$x_{ij}$。
$$
\begin{array}{l}
x_{i j}^{\prime}=\frac{x_{i j}-\min \left\{x_{1 j}, \ldots, x_{n j}\right\}}{\max \left\{x_{1 j}, \ldots, x_{r j}\right\}-\min \left\{x_{1 j}, \ldots, x_{n j}\right\}}
\end{array}
$$
$$
\begin{array}{l}x_{i j}^{\prime}=\frac{\max \left\{x_{1 j}, \ldots, x_{n j}\right\}-x_{i j}}{\max \left\{x_{1 j}, \ldots, x_{n j}\right\}-\min \left\{x_{1 j}, \ldots, x_{n j}\right\}}
\end{array}
$$

3. 计算第$j$项指标下第$i$个样本值占该指标的比重：
$$
\begin{array}{l}p_{i j}=\frac{x_{i j}}{\sum_{i=1}^{n} x_{i j}}, \quad i=1, \cdots, n, j=1, \cdots, m
\end{array}
$$

4. 计算第$j$项指标的熵值：
   - 其中，$k=\frac{1}{ln(n)}>0$ 满足$e{j} \ge 0$;
$$
\begin{array}{l}e_{j}=-k \sum_{i=1}^{n} p_{i j} \ln \left(p_{i j}\right), \quad j=1, \cdots, m
\end{array}
$$

5. 计算信息熵冗余度（差异）：
$$
\begin{array}{l}d_{j}=1-e_{j}, \quad j=1, \cdots, m
\end{array}
$$

6. 计算各项指标的权重：
$$
\begin{array}{l}d_{j}=1-e_{j}, \quad j=1, \cdots, mw_{j}=\frac{d_{j}}{\sum_{j=1}^{m} d_{j}}, \quad j=1, \cdots, m
\end{array}
$$

7. 计算各样本的综合得分：
   - 其中，$x_{ij}$为标准化后的数据。
$$
\begin{array}{l}s_{i}=\sum_{j=1}^{m} w_{j} x_{i j}, \quad i=1, \cdots, n
\end{array}
$$