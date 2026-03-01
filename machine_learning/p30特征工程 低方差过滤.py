from sklearn.feature_selection import VarianceThreshold


import numpy as np
a = np.random.randn(100)
# 生成100个数，平均在1
# print(a)
print(np.var(a))

# 平均5 标准差0.1
b = np.random.normal(5,0.1,size = 100)
# 方差
print(np.var(b))
# 平均值
print(np.mean(b))


X = np.vstack((a, b)).T
print(X,X.shape)

vt = VarianceThreshold(0.01)
X_filtered = vt.fit_transform(X)
print(X_filtered)



