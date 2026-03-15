import numpy as np
import matplotlib.pyplot as plt



# 定义目标函数
def f(x):
    return (x ** 2 - 2) ** 2
# 定义梯度函数（导函数）
def gradient(x):
    return 4 * x ** 3 - 8 * x


# 用列表保存点的变化轨迹
x_list = []
y_list = []

# 定义超参数和x的初始值
alpha = 0.1
x = 1

for i in range(100):
    y = f(x)
    # 记录当前点的坐标
    x_list.append(x)
    y_list.append(y)
    print(f"x={x}, f(x)={y}")
    # 计算梯度
    grad = gradient(x)
    # 更新参数
    x = x - alpha * grad

# 画图
# x = np.arange(0.9, 1.6, 0.01)
# plt.plot(x, f(x))
# plt.plot(x_list, y_list, "r")
# plt.scatter(x_list, y_list, color="r")
# plt.show()



# 局部放大
fig, ax = plt.subplots(1, 2, figsize=(15, 4))
ax[0].plot(x, f(x))
ax[0].plot(x_list, y_list, "r")
ax[0].scatter(x_list, y_list, color="r")

x_list2 = x_list[1:]
y_list2 = y_list[1:]
x = np.arange(1.399, 1.425, 0.001)
ax[1].plot(x, f(x))
ax[1].plot(x_list2, y_list2, "r")
ax[1].scatter(x_list2, y_list2, color="r")
plt.show()