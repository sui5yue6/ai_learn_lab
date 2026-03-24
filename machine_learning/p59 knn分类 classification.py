from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import matplotlib.pyplot as plt

# 准备数据
X = np.array([[2, 1], [3, 1], [1, 4], [2, 6]])
y = np.array([0, 1, 0, 1])  # 分类标签

# 定义KNN分类模型
knn = KNeighborsClassifier(n_neighbors=2, weights='distance')

# 模型训练
knn.fit(X, y)

# 预测
x = np.array([[4, 9]])
x_class = knn.predict(x)
print(x_class)

# 画图
fig, ax = plt.subplots()
ax.axis('equal')
# 使用布尔索引将两类点分开
X1 = X[y == 0]
X2 = X[y == 1]

print('X1', X1, X1[:, 0], X1[:, 1])
print('X2', X2, X2[:, 0], X2[:, 1])

# 定义不同的颜色，画两组点
colors = ["C0", "C1"]
plt.scatter(X1[:, 0], X1[:, 1], c=colors[0])
plt.scatter(X2[:, 0], X2[:, 1], c=colors[1])
# 新点的颜色
x_color = colors[0] if x_class == 0 else colors[1]
plt.scatter(x[:, 0], x[:, 1], c=x_color)
plt.show()
