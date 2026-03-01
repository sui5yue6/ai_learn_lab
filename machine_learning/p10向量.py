import numpy as np

x = np.array([1, 2, 3, 4, 5])
print(x)
print(x.shape)

print(x.T)
print(x.T.shape)
y = np.array([1, 2, 3, 4, 5])

print(x * y)
print(x.dot(y))  # 这是是点积，不是矩阵的相乘
# 这里dot再补充一下，又是点积，又是矩阵乘法的，这样定义很容易让人疑惑
print(x @ y)

# 计算范数
l0 = np.linalg.norm(x, ord=0)
print(l0)  # 计算范数
l1 = np.linalg.norm(x, ord=1)
print(l1)
l2 = np.linalg.norm(x, ord=2)
print(l2)
