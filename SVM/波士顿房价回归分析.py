from sklearn.datasets import load_boston
boston = load_boston()
#打印数据集中的键
print(boston.keys())

from sklearn.model_selection import train_test_split
X, y = boston.data, boston.target
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 8)
print(X_train.shape)
print(X_test.shape)

from sklearn.svm import SVR
#分别测试linear核函数和rbf核函数
for kernel in ['linear', 'rbf']:
    svr = SVR(kernel = kernel)
    svr.fit(X_train, y_train)
    print(kernel, '核函数的模型训练集得分：{:.3f}'.format(
        svr.score(X_train, y_train)))
    print(kernel, '核函数的模型测试集得分：{:.3f}'.format(
        svr.score(X_test, y_test)))

import matplotlib.pyplot as plt
#将特征数值中的最小值和最大值用散点画出来
plt.plot(X.min(axis = 0), 'v', label = 'min')
plt.plot(X.max(axis = 0), 'v', label = 'max')
#设定纵坐标为对数形式
plt.yscale('log')
#设置图注位置为最佳
plt.legend(loc = 'best')
plt.xlabel('features')
plt.ylabel('feature magnitude')
plt.show()

#导入数据预处理工具
from sklearn.preprocessing import StandardScaler
#对训练集和测试集进行预处理
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

#将预处理后的数据特征最大值和最小值用散点图表示出来
plt.plot(X_train_scaled.min(axis = 0), 'v', label = 'train set min')
plt.plot(X_train_scaled.max(axis = 0), 'v', label = 'train set min')
plt.plot(X_test_scaled.min(axis = 0), 'v', label = 'test set min')
plt.plot(X_test_scaled.max(axis = 0), 'v', label = 'test set min')
plt.yscale('log')

#设置图注位置
plt.legend(loc = 'best')
plt.xlabel('scaled features')
plt.ylabel('scaled feature magnitude')
plt.show()

#用预处理后的数据重新训练模型
for kernel in ['linear', 'rbf']:
    svr = SVR(kernel = kernel)
    svr.fit(X_train_scaled, y_train)
    print('数据预处理后', kernel, '核函数的模型训练集得分：{:.3f}'.format(
        svr.score(X_train_scaled, y_train)))
    print('数据预处理后', kernel, '核函数的模型测试集得分：{:.3f}'.format(
        svr.score(X_test_scaled, y_test)))
    

#设置模型的C参数和gamma参数
svr = SVR(C = 100, gamma = 0.1)
svr.fit(X_train_scaled, y_train)
print('调节参数后的模型在训练集的得分：{:.3f}'.format(
        svr.score(X_train_scaled, y_train)))
print('调节参数后的模型在测试集的得分：{:.3f}'.format(
        svr.score(X_test_scaled, y_test)))
