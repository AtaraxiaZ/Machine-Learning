from sklearn.datasets import load_wine

#从sklearn的datashts载入数据集
wine_dataset = load_wine()

#load_wine函数载入的数据集是Bunch对象，包括keys和values
print('红酒数据集中的键:\n{}'.format(wine_dataset.keys()))

#可以用.shape语句来让python告诉我们数据的大概轮廓
#返回结果第一个是样本数量，第二个是每条数据含有的特征变量数
print('数据概况:{}'.format(wine_dataset['data'].shape))

#更详细的细节可以通过打印DESCR键来获得
print(wine_dataset['DESCR'])

#导入数据集拆分工具
from sklearn.model_selection import train_test_split

#将数据集拆分为训练数据集合测试数据集，伪随机数设为缺省
X_train, X_test, y_train, y_test = train_test_split(
    wine_dataset['data'], wine_dataset['target'], random_state = 0)

#查看拆分后的数据集
#前两个是特征向量的形态，后两个是目标（标签）的形态
print('X_train shape:{}'.format(X_train.shape))
print('X_test shape:{}'.format(X_test.shape))
print('y_train shape:{}'.format(y_train.shape))
print('y_test shape:{}'.format(y_test.shape))

#使用K最邻近算法进行建模
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 1)

#用模型对数据进行拟合，依据就是训练数据集中的样本数据及其对应标签
#knn的拟合方法把自身作为结果返回
#从结果中可以看到模型的全部参数设定（这些参数用默认值即可）
knn.fit(X_train, y_train)
print(knn)

#用测试数据集给模型打分。吻合度越高，分数越高，最高1.0
print('测试数据集得分：{:.2f}'.format(knn.score(X_test, y_test)))

#用建好的模型对新酒做出分类预测，虽然精确率76%很低了
import numpy as np
X_new = np.array([[13.2, 2.77, 2.51, 18.5, 96.6, 1.04, 2.55, 0.57,
                   1.47, 6.2, 1.05, 3.33, 820]])
#使用.predict进行预测
prediction = knn.predict(X_new)
print('预测新红酒的分类是：{}'.format(wine_dataset['target_names'][prediction]))


