import numpy as np
import matplotlib.pyplot as plt
#令x为-5到5之间，元素书为100的等差数列
x = np.linspace(-5, 5, 100)
#输入直线方程
y = 0.5*x + 3
plt.plot(x, y, c = 'orange')
plt.title('Straight Line')
plt.show()

#导入线性回归模型，用自己设置的点来设定直线
from sklearn.linear_model import LinearRegression
#输入两个点的横坐标和纵坐标
X = [[1], [4]]
y = [3, 5]
#用线性模型拟合这两个点
lr = LinearRegression().fit(X, y)
#画出两个点和直线的图形
z = np.linspace(0, 5, 20)
plt.scatter(X, y, s = 80)
plt.plot(z, lr.predict(z.reshape(-1,1)), c = 'k')
plt.title('Straight Line')
plt.show()

#打印直线方程，coef_和intercept_的下划线表示来自训练数据集的属性
print('y = {:.3f}'.format(lr.coef_[0]), 'x',' + {:.3f}'.format(lr.intercept_))

#3个点的情况
X = [[1], [4], [3]]
y = [3, 5, 3]
#用线性模型拟合这3个点
lr = LinearRegression().fit(X, y)
#画出2个点和直线的图形
z = np.linspace(0, 5, 20)
plt.scatter(X, y, s = 80)
plt.plot(z, lr.predict(z.reshape(-1,1)), c = 'k')
plt.title('Straight Line')
plt.show()

print('y = {:.3f}'.format(lr.coef_[0]), 'x',' + {:.3f}'.format(lr.intercept_))

from sklearn.datasets import make_regression
#生成用于回归分析的数据集
X, y = make_regression(n_samples = 50, n_features = 1, n_informative = 1,
                       noise = 50, random_state = 1)
#使用线性模型对数据进行拟合
reg = LinearRegression()
reg.fit(X, y)
#z是我们生成的等差数列，用来画出线性模型的图形
z = np.linspace(-3, 3, 200).reshape(-1,1)
plt.scatter(X, y, c = 'b', s = 60)
plt.plot(z, reg.predict(z), c = 'k')
plt.title('Liner Regression')

print('y = {:.3f}'.format(reg.coef_[0]), 'x',' + {:.3f}'.format(reg.intercept_))


