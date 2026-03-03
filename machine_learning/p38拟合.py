import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression  # 线性回归模型
from sklearn.preprocessing import PolynomialFeatures  # 构建多项式特征
from sklearn.model_selection import train_test_split  # 划分训练集和测试集
from sklearn.metrics import mean_squared_error  # 均方误差损失函数

# ========== 核心配置：解决中文显示问题 ==========
plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题   不使用unicode来显示

X = np.linspace(-3, 3, 300).reshape(-1, 1)
# 这里是让其在-3到3之间，均值分布

# reshape(-1, 1)是让他排成一列
# print(np.linspace(-3, 3, 300))
print('X', X)

y = np.sin(X) + np.random.uniform(low=-0.5, high=0.5, size=300).reshape(-1, 1)
print('y', y)

fig, ax = plt.subplots(1, 3, figsize=(15, 4))
ax[0].scatter(X, y, c='y')
ax[1].scatter(X, y, c='y')
ax[2].scatter(X, y, c='y')
# plt.show() # 放下面在绘制
trainX, testX, trainY, testY = train_test_split(X, y, test_size=0.2, random_state=42)
# random_state=42 随机种子


model = LinearRegression()
model.fit(trainX, trainY)
# 因为是linear的，因此可以直接查看斜率
print(model.coef_, model.intercept_)
y_pred1 = model.predict(testX)
# 画出拟合曲线


test_loss1 = mean_squared_error(testY, y_pred1)  #
train_loss1 = mean_squared_error(trainY, model.predict(trainX))
print('test_loss1  测试的标准答案和预测的答案的误差', test_loss1)
print('train_loss1  训练的标准答案和预测的答案的误差', train_loss1)
# train_loss1这个数值大 这个是欠拟合的。等过拟合的时候，这个数值就会很接近

# 画出拟合曲线，并写出训练误差和测试误差
ax[0].plot(X, model.predict(X), 'r')  # 这里的x是随机的，但是x和y组成的关系是线性的
ax[0].text(-3, 1, f"测试误差：{test_loss1:.4f}")
ax[0].text(-3, 1.3, f"训练误差：{train_loss1:.4f}")
# 折柳的-3，1 -3,1.3 代表是文字放置在坐标轴的什么位置


# 正好拟合，使用5次多项式
poly5 = PolynomialFeatures(degree=5)
x_train2 = poly5.fit_transform(trainX)
x_test2 = poly5.fit_transform(testX)
print(x_train2.shape, x_train2)
# 生成的特征名称（简化）： ['1' 'x1' 'x1^2' 'x1^3' 'x1^4' 'x1^5']
print(x_test2.shape)

model.fit(x_train2, trainY)
y_pred2 = model.predict(x_test2)
test_loss2 = mean_squared_error(testY, y_pred2)
train_loss2 = mean_squared_error(trainY, model.predict(x_train2))

# 画出拟合曲线，并写出训练误差和测试误差
ax[1].plot(X, model.predict(poly5.fit_transform(X)), 'r')
ax[1].text(-3, 1, f"测试误差：{test_loss2:.4f}")
ax[1].text(-3, 1.3, f"训练误差：{train_loss2:.4f}")

poly20 = PolynomialFeatures(degree=20)
x_train3 = poly20.fit_transform(trainX)
x_test3 = poly20.fit_transform(testX)
print(x_train3.shape)
print(x_test3.shape)
model.fit(x_train3, trainY)
y_test_pred3 = model.predict(x_test3)
test_loss3 = mean_squared_error(testY, y_test_pred3)
y_train_pred3 = model.predict(x_train3)
train_loss3 = mean_squared_error(trainY, y_train_pred3)

# 画出拟合曲线，并写出训练误差和测试误差
ax[2].plot(X, model.predict(poly20.fit_transform(X)), 'r')
ax[2].text(-3, 1, f"测试误差：{test_loss3:.4f}")
ax[2].text(-3, 1.3, f"训练误差：{train_loss3:.4f}")

plt.show()


"""
三个图
训练误差在减小
但是测试误差，先是减小，然后增大
"""