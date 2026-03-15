import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Lasso, Ridge  # 线性回归模型
from sklearn.preprocessing import PolynomialFeatures  # 构建多项式特征
from sklearn.model_selection import train_test_split  # 划分训练集和测试集
from sklearn.metrics import mean_squared_error  # 均方误差损失函数

# ========== 核心配置：解决中文显示问题 ==========
plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题   不使用unicode来显示

X = np.linspace(-3, 3, 300).reshape(-1, 1)
# 这里是让其在-3到3之间，均值分布
y = np.sin(X) + np.random.uniform(low=-0.5, high=0.5, size=300).reshape(-1, 1)

fig, ax = plt.subplots(2, 3, figsize=(15, 8))
ax[0,0].scatter(X, y, c='y')
ax[0,1].scatter(X, y, c='y')
ax[0,2].scatter(X, y, c='y')
# plt.show() # 放下面在绘制
trainX, testX, trainY, testY = train_test_split(X, y, test_size=0.2, random_state=42)

# 过拟合情况（20次多项式）
poly20 = PolynomialFeatures(degree=20)
x_train = poly20.fit_transform(trainX)
x_test = poly20.fit_transform(testX)
print(x_train.shape)
print(x_test.shape)

# 一、不加正则化项

# 定义模型
model = LinearRegression()

# 4. 训练模型
model.fit(x_train, trainY)

# 5. 预测结果，计算误差
y_pred1 = model.predict(x_test)
test_loss1 = mean_squared_error(testY, y_pred1)

# 画出拟合曲线，并写出训练误差和测试误差
ax[0,0].plot(X, model.predict(poly20.fit_transform(X)), 'r')
ax[0,0].text(-3, 1, f"测试误差：{test_loss1:.4f}")
# 画所有系数的直方图
ax[1,0].bar(np.arange(21), model.coef_.reshape(-1))

# 二、加L1正则化项（Lasso回归）

# 定义模型
lasso = Lasso(alpha=0.01)

# 4. 训练模型
lasso.fit(x_train, trainY)

# 5. 预测结果，计算误差
y_pred2 = lasso.predict(x_test)
test_loss2 = mean_squared_error(testY, y_pred2)

# 画出拟合曲线，并写出训练误差和测试误差
ax[0,1].plot(X, lasso.predict(poly20.fit_transform(X)), 'r')
ax[0,1].text(-3, 1, f"测试误差：{test_loss2:.4f}")
# 画所有系数的直方图
ax[1,1].bar(np.arange(21), lasso.coef_.reshape(-1))

# 三、加L2正则化项（岭回归）

# 定义模型
ridge = Ridge(alpha=1)

# 4. 训练模型
ridge.fit(x_train, trainY)

# 5. 预测结果，计算误差
y_pred3 = ridge.predict(x_test)
test_loss3 = mean_squared_error(testY, y_pred3)

# 画出拟合曲线，并写出训练误差和测试误差
ax[0,2].plot(X, ridge.predict(poly20.fit_transform(X)), 'r')
ax[0,2].text(-3, 1, f"测试误差：{test_loss3:.4f}")
# 画所有系数的直方图
ax[1,2].bar(np.arange(21), ridge.coef_.reshape(-1))

plt.show()
#  可以看出 L1的正则化，只留下了部分的系数



