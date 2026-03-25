import numpy as np
from sklearn.linear_model import LinearRegression


# 1. 定义数据
# 自变量，每周学习时长
X = [[5], [8], [10], [12], [15], [3], [7], [9], [14], [6]]
# 因变量，数学考试成绩
y = [55, 65, 70, 75, 85, 50, 60, 72, 80, 58]

# 2. 创建模型：线性回归
model = LinearRegression()


# 3. 模型训练
model.fit(X, y)

# 打印模型参数
print(model.coef_)
print(model.intercept_)


# 用方差和协方差验证数学求解公式
x = np.array(X).reshape(-1)
cov = np.cov(x, y)
print(cov)

beta1 = cov[0][1] / cov[0][0]
# 斜率
print(beta1)
# 截距
print(np.mean(y) - beta1 * np.mean(x))






model = LinearRegression(fit_intercept=False)
model.fit(X, y)



