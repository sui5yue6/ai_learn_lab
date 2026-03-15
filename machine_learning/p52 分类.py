#%%
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns


# 定义类别标签
labels = ["猫", "狗"]
# 定义数据（预测值和真实值）
y_true = ["猫", "猫", "猫", "猫", "猫", "猫", "狗", "狗", "狗", "狗"]
y_pred = ["猫", "猫", "狗", "猫", "猫", "猫", "猫", "猫", "狗", "狗"]


# 得到混淆矩阵
matrix = confusion_matrix(y_true, y_pred, labels=labels)
print(matrix)

print(pd.DataFrame(matrix, columns=labels, index=labels))



sns.heatmap(matrix, annot=True, fmt='d', cmap="Greens")
plt.show()


from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_true, y_pred)
print(accuracy)

from sklearn.metrics import precision_score
precision = precision_score(y_true, y_pred, pos_label="猫")
print(precision)

from sklearn.metrics import recall_score
recall = recall_score(y_true, y_pred, pos_label="猫")
print(recall)

from sklearn.metrics import f1_score
f1 = f1_score(y_true, y_pred, pos_label="猫")
print(f1)

from sklearn.metrics import classification_report
report = classification_report(y_true, y_pred, labels=labels, target_names=None)
print(report)