import numpy as np

f = np.array([1, 2, 4, 7, 11, 16])
grad = np.gradient(f)
print(grad)
A = np.array([[1, 2, 3],
              [4, 5, 6]])

grad = np.gradient(A)
print(grad)
