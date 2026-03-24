import numpy as np
from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt

# 准备数据
X = [[2, 1], [3, 1], [1, 4], [2, 6]]
y = [0.5, 0.33, 4, 3]    # 分类标签

# KNN回归模型
knn = KNeighborsRegressor(n_neighbors=2, weights='distance')
knn.fit(X, y)
# 预测
x = [[4, 9]]
x_pred = knn.predict(x)
print(x_pred)


# print('X1', X, X[:, 0], X[:, 1])



# 画图
fig, ax = plt.subplots()
ax.axis('equal')
# 使用布尔索引将两类点分开
# X1 = X[y == 0]
# X2 = X[y == 1]
# 定义不同的颜色，画两组点
colors = ["C0", "C1"]
plt.scatter(np.array(X)[:, 0], np.array(X)[:, 1], c=colors[0])
# plt.scatter(X2[:, 0], X2[:, 1], c=colors[1])
# 新点的颜色
# x_color = colors[0] if x_class == 0 else colors[1]
# plt.scatter(x[:, 0], x[:, 1], c=x_color)
plt.show()