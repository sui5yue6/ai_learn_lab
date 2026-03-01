import pandas as pd

advertising = pd.read_csv('data/advertising.csv')
# print(advertising.head())
# print(advertising.describe())
# print(advertising.shape)

advertising.drop(advertising.columns[0], axis=1,
                 inplace=True  # 直接修改原数据对象 而不是创建一个新的、修改后的副本返回。
                 )
# 去掉空值
advertising.dropna(inplace=True)

print(advertising.head())
print(advertising.describe())
print(advertising.shape)
# 提取特征和标签（目标值）
X = advertising.drop("Sales",
                     axis=1  # 表示维度是列
                     )
y = advertising["Sales"]

# 计算皮尔逊相关系数
print(X.corrwith(y, method="pearson"))

# X内部，所有列的相关系数
corr_matrix = advertising.corr(method="pearson")
print(corr_matrix)


# 将相关系数矩阵画成热力图
import seaborn as sns
import matplotlib.pyplot as plt
# coolwarm冷热图
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Matrix")
plt.show()