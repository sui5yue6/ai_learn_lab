import sklearn
from sklearn.linear_model import LogisticRegression
print(sklearn.__version__)


model = LogisticRegression(
    solver='sag',
    # multi_class='multinomial',
    max_iter=1000,
    class_weight='balanced',
    random_state=42,
    penalty='l1',
    C=1.0

)

# OVR
# 1. 直接创建LogisticRegression模型
model_ovr1 = LogisticRegression(
    # multi_class='ovr'
)

from sklearn.multiclass import OneVsRestClassifier

# 2. 创建OneVsRestClassifier模型
model_ovr2 = OneVsRestClassifier(LogisticRegression())

# 就是说,默认就是softmax吧
model_softmax = LogisticRegression(
    # multi_class='multinomial'
)

# print(model.pr)
print(model)
print(model_ovr1)
print(model_ovr2)
print(model_softmax)

