import numpy as np

A = np.array([[1, 2, 3],
              [4, 5, 6]])
B = np.array([[1, 3, 7],
              [5, 0, 2]])

print(A)
print(A.shape)

# 矩阵和向量
x = np.array([1, 3, 7])
print(A * x)

# 矩阵乘以矩阵 这里的写法让人很疑惑。哈达玛积。
print(A * B)

# 矩阵乘法
print(A.dot(B.T))
print(A @ B.T)
C = np.linalg.pinv(A)
print('C', C)
print('A@C', A @ C)

A = np.array([[1, 2],
              [3, 5]])
C = np.linalg.inv(A)
print(C)
C = np.around(C).astype(int)
print(C)
print('A@C', A @ C)
"""
 矩阵求逆
"""
