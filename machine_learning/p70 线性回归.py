
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

# 5. 预测
x_new = [[11]]
y_pred = model.predict(x_new)
print(y_pred)


# 6. 画图
import matplotlib.pyplot as plt
import numpy as np
# 将竖的变成行的
x_line = np.arange(0, 15, 0.1).reshape(-1, 1)
y_line = model.predict(x_line)
plt.scatter(X, y, color='black')
plt.plot(x_line, y_line, color='red')
plt.scatter(x_new, y_pred, color='green')
plt.show()