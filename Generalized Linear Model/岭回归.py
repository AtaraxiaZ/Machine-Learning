from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
#载入糖尿病情数据集
X, y = load_diabetes().data, load_diabetes().target
#将数据源拆分成训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 8)
lr = LinearRegression().fit(X_train, y_train)

print('标准线性回归训练数据集得分：{:.2f}'.format(lr.score(X_train, y_train)))
print('标准线性回归测试数据集得分：{:.2f}'.format(lr.score(X_test, y_test)))


from sklearn.linear_model import Ridge
ridge = Ridge().fit(X_train, y_train)
print('默认ridge1训练数据集得分：{:.2f}'.format(ridge.score(X_train, y_train)))
print('默认ridge1测试数据集得分：{:.2f}'.format(ridge.score(X_test, y_test)))


ridge10 = Ridge(alpha = 10).fit(X_train, y_train)
print('ridge10训练数据集得分：{:.2f}'.format(ridge10.score(X_train, y_train)))
print('ridge10测试数据集得分：{:.2f}'.format(ridge10.score(X_test, y_test)))

ridge01 = Ridge(alpha = 0.1).fit(X_train, y_train)
print('ridge01训练数据集得分：{:.2f}'.format(ridge01.score(X_train, y_train)))
print('ridge01测试数据集得分：{:.2f}'.format(ridge01.score(X_test, y_test)))


import matplotlib.pyplot as plt
'''
#绘制alpha=1时的模型系数
plt.plot(ridge.coef_, 's', label = 'Ridge alpha=1')
#绘制alpha=10时的模型系数
plt.plot(ridge10.coef_, '^', label = 'Ridge alpha=10')
#绘制alpha=0.1时的模型系数
plt.plot(ridge01.coef_, 'v', label = 'Ridge alpha=0.1')
#绘制线性回归的系数作为对比
plt.plot(lr.coef_, 'o', label = 'liner regression')
#横坐标是系数序号
plt.xlabel('coefficient index')
#纵坐标是系数量级
plt.ylabel('coefficient magnitude')
plt.hlines(0, 0, len(lr.coef_))
plt.legend()
'''

import numpy as np
from sklearn.model_selection import learning_curve, KFold
#定义一个绘制学习曲线的函数
def plot_learning_curve(est, X, y):
#将数据进行20次拆分来对模型进行评分
    #est，estimator估计量
    training_set_size, train_scores, test_scores = learning_curve(
        est, X, y, train_sizes = np.linspace(.1, 1, 20), cv = KFold(20, shuffle = True,
                                                                    random_state = 1))
    estimator_name = est.__class__.__name__
    line  = plt.plot(training_set_size, train_scores.mean(axis = 1), '--',
                     label = 'training ' + estimator_name)
    plt.plot(training_set_size, test_scores.mean(axis = 1), '-',
             label = 'test ' + estimator_name, c = line[0].get_color())
    plt.xlabel('Training set size')
    plt.ylabel('Score')
    plt.ylim(0, 1.1)

plot_learning_curve(Ridge(alpha = 1), X, y)
plot_learning_curve(LinearRegression(), X, y)
plt.legend(loc = (0, 1.05), ncol = 2, fontsize = 11)
