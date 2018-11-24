import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso


X, y = load_diabetes().data, load_diabetes().target
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 8)

#使用套索回归拟合数据
lasso = Lasso().fit(X_train, y_train)
print("套索回归在训练数据集中的得分：{:.2f}".format(lasso.score(X_train, y_train)))
print("套索回归在测试数据集中的得分：{:.2f}".format(lasso.score(X_test, y_test)))
print("套索回归使用的特征数：{}".format(np.sum(lasso.coef_ != 0)))

#增加最大迭代次数max_iter的默认设置
#否则模型会提示我们增加最大迭代次数
lasso01 = Lasso(alpha = 0.1, max_iter = 100000).fit(X_train, y_train)
print("alpha01在训练数据集中的得分：{:.2f}".format(lasso01.score(X_train, y_train)))
print("alpha01在测试数据集中的得分：{:.2f}".format(lasso01.score(X_test, y_test)))
print("alpha01使用的特征数：{}".format(np.sum(lasso01.coef_ != 0)))

lasso00001 = Lasso(alpha = 0.0001, max_iter = 100000).fit(X_train, y_train)
print("alpha00001在训练数据集中的得分：{:.2f}".format(lasso00001.score(X_train, y_train)))
print("alpha00001在测试数据集中的得分：{:.2f}".format(lasso00001.score(X_test, y_test)))
print("alpha00001使用的特征数：{}".format(np.sum(lasso01.coef_ != 0)))

import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
ridge01 = Ridge(alpha = 0.1).fit(X_train, y_train)
plt.plot(lasso.coef_, 's', label = 'Lasso alpha = 1')
plt.plot(lasso01.coef_, '^', label = 'Lasso alpha = 0.1')
plt.plot(lasso00001.coef_, 'v', label = 'Lasso alpha = 0.0001')
plt.plot(ridge01.coef_, 'o', label = 'Ridge alpha = 0.1')
plt.legend(ncol = 2, loc = (0, 1.05))
plt.ylim(-800, 800)
plt.xlabel("Coefficient index")
plt.ylabel("Coefficient magnitude")
