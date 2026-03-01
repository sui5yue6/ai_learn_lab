import numpy as np

A = np.array([[1, 2, 3],
              [4, 5, 6]])
B = np.array([[1, 3, 7],
              [5, 0, 2]])
print(A * B)
print(np.multiply(A, B))

print(np.kron(A, B))
