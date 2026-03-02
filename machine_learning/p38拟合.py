import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression   # 线性回归模型
from sklearn.preprocessing import PolynomialFeatures    # 构建多项式特征
from sklearn.model_selection import train_test_split    # 划分训练集和测试集
from sklearn.metrics import mean_squared_error  # 均方误差损失函数

X = np.linspace(-3, 3, 300).reshape(-1, 1)
# reshape(-1, 1)是让他排成一列
# print(np.linspace(-3, 3, 300))
print(X)

y = np.sin(X) + np.random.uniform(low=-0.5, high=0.5, size=300).reshape(-1, 1)
print(y)

fig, ax = plt.subplots(1, 3, figsize=(15, 4))
ax[0].scatter(X, y, c='y')
ax[1].scatter(X, y, c='y')
ax[2].scatter(X, y, c='y')
plt.show()
trainX, testX, trainY, testY = train_test_split(X, y, test_size=0.2, random_state=42)
# random_state=42 随机种子


model = LinearRegression()

