import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split

cancer = load_breast_cancer()
print(cancer.keys())
print('肿瘤的分类：', cancer['target_names'])
print('肿瘤的特征：', cancer['feature_names'])

#拆分的语法
X, y = cancer.data, cancer.target
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 38)

#输出的是样本数量和每个样本中含有的特征数量
print('训练集数据状态：', X_train.shape)
print('测试记数据状态：', X_test.shape)


gnb = GaussianNB()
gnb.fit(X_train, y_train)
print('模型得分：{:.3f}'.format(gnb.score(X_test, y_test)))

#随机使用一个样本让模型进行预测，看是否可以分到正确的类中
print('模型预测的分类是：{}'.format(gnb.predict([X[312]])))
print('样本的正确分类是：', y[312])


import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
#导入随机拆分工具
from sklearn.model_selection import ShuffleSplit
#定义一个函数绘制学习曲线
def plot_learning_curve(estimator, title, X, y, ylim = None, cv = None,
                        n_jobs = 1, train_sizes = np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    #设定横轴标签
    plt.xlabel('Training examples')
    plt.ylabel('Score')
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv = cv, n_jobs = n_jobs, train_sizes = train_sizes)
    train_scores_mean = np.mean(train_scores, axis = 1)
    test_scores_mean = np.mean(test_scores, axis = 1)
    plt.grid()
    plt.plot(train_sizes, train_scores_mean, 'o-', color = 'r',
             label = 'Training score')
    plt.plot(train_sizes, test_scores_mean, 'o-', color = 'g',
             label = 'Cross-validation score')
    plt.legend(loc = 'lower right')
    return plt


title = 'Learning Curves (Naive Bayes)'
#设定拆分数量
cv = ShuffleSplit(n_splits = 100, test_size = 0.2, random_state = 0)
#设定模型为高斯朴素贝叶斯
estimator = GaussianNB()

plot_learning_curve(estimator, title, X, y, ylim = (0.9, 1.01),
                    cv = cv, n_jobs = 4)
plt.show()



