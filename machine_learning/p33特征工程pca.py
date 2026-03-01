import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

X = np.random.randn(1000, 3)
# print(X.shape)
# print(X)

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



