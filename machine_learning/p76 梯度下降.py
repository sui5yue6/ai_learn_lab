import numpy as np
from sklearn.linear_model import LinearRegression

# 定义损失函数
def J(beta):
    return np.sum( (X @ beta - y) ** 2 ) / n
# 均方误差（MSE）损失函数

# 就是J的求导
# 定义计算梯度的函数
def gradient(beta):
    return X.T @ (X @ beta - y) / n * 2


# 1. 定义数据
X = np.array([[5], [8], [10], [12], [15], [3], [7], [9], [14], [6]])  # 自变量，每周学习时长
y = np.array([[55], [65], [70], [75], [85], [50], [60], [72], [80], [58]])


n = X.shape[0]

# 2. 数据处理，X增加一列1
X = np.hstack((np.ones((n, 1)), X))
"""
[[ 1.  5.], [ 1.  8.], [ 1. 10.], [ 1. 12.], [ 1. 15.], [ 1.  3.], [ 1.  7.], [ 1.  9.], [ 1. 14.], [ 1.  6.]]
"""
print(X)

# 3. 初始化参数以及超参数
alpha = 0.01
iter = 10000

beta = np.array([[1], [1]])

# 定义列表，保存参数变化轨迹
beta0 = []
beta1 = []

# 重复迭代
# for i in range(iter):
while (np.abs(grad := gradient(beta)) > 1e-10).any() and (iter := iter - 1) >= 0:
    beta0.append(beta[0, 0])
    beta1.append(beta[1, 0])

    # 4. 计算梯度
    # grad = gradient(beta)

    # 5. 更新参数
    beta = beta - alpha * grad

    # 每迭代10轮打印一次当前的参数值和损失值
    if iter % 10 == 0:
        print(f"beta: {beta.reshape(-1)}\tJ: {J(beta)}")

print(iter)

# 画图
import matplotlib.pyplot as plt
plt.plot(beta0, beta1)
plt.show()

# 最终的结果,就是上一个文件的
"""
当前文件计算值
beta: [41.45069394  2.87070855]	J: 2.9811541271000688

上一个文件计算值
2.8707085463842223
41.45069393718042
"""