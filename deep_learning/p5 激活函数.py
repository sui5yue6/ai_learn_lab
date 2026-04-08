# 阶跃函数
def step_function0(x):
    if x > 0:
        return 1
    else:
        return 0


import numpy as np


# 阶跃函数
def step_function(x):
    return np.array(x > 0, dtype=int)


# Sigmoid函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# ReLU函数
def relu(x):
    return np.maximum(0, x)


# Softmax函数
def softmax0(x):
    return np.exp(x) / np.sum(np.exp(x))


# 考虑输入可能是矩阵的情况
def softmax(x):
    # 如果是二维矩阵
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T
    # 溢出处理策略
    x = x - np.max(x)
    return np.exp(x) / np.sum(np.exp(x))


if __name__ == '__main__':
    x = np.array([0, 1, 2, 3, 4, 5, -1, -2, -3, -4, -5])
    print(step_function(x))
    print(sigmoid(x))
    # 双曲正切函数
    print(np.tanh(x))
    # 双曲正切函数容易产生梯度消失

    # 定义简单，不会存在梯度消失， 只有部分是活跃的，带来稀疏性
    # 缺点是，导致神经元死亡。因此有 leaky reLU进行补丁
    print(relu(x))


    print(softmax0(x))
    X = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [-1, -2, -3]])
    print(softmax(X))
