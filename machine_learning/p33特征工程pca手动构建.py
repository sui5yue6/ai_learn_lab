import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

n = 1000
# 定义两个主成分方向向量
pc1 = np.random.normal(0, 10, n)
pc2 = np.random.normal(0, 2, n)

# pc1 = np.random.normal(0, 1, n)
# pc2 = np.random.normal(0, 0.2, n)
# 定义不重要的第三主成分（噪声）
noise = np.random.normal(0, 0.05, n)
# 构建3个特征的输入数据X
X = np.vstack((pc1 + pc2, pc1 - pc2, pc2 + noise)).T


# 使用PCA进行降维，将3维数据降为2维
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
print(X_pca.shape)


# 可视化
# 转换前的3维数据可视化
fig = plt.figure(figsize=(12, 4))
ax1 = fig.add_subplot(121, projection='3d')
# 所有行，第0列， 所有行，第1列，所有行，第2列
# c=g表示点是绿色的
ax1.scatter(X[:, 0], X[:, 1], X[:, 2], c="green")
ax1.set_title('Before PCA(3D)')
ax1.set_xlabel('Feature1')
ax1.set_ylabel('Feature2')
ax1.set_zlabel('Feature3')
# 转换后的2维数据可视化
ax2 = fig.add_subplot(122)
ax2.scatter(X_pca[:, 0], X_pca[:, 1], c="g")
ax2.set_title('After PCA(2D)')
ax2.set_xlabel('Principal Component1')
ax2.set_ylabel('Principal Component2')
plt.show()

# 这里可以打印看下，维度的值的数量冀
print(X)
print(X_pca)

# 看二维的Component1 Component2 可以看出是两个数量冀