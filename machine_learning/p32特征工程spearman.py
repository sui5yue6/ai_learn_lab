import pandas as pd

"""
排名计算相关性
"""

# 定义数据
X = [[5], [8], [10], [12], [15], [3], [7], [9], [14], [6]]
y = [55, 65, 70, 75, 85, 50, 60, 72, 80, 58]

X = pd.DataFrame(X)
y = pd.Series(y)

print(X.shape)
print(y.shape)
print(X.corrwith(y, method="spearman"))
