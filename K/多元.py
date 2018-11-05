from sklearn.datasets import make_blobs
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
#生成样本数为500，分类为5的数据集
data = make_blobs(n_samples = 500, centers = 5, random_state = 8)
X, y = data
plt.scatter(X[:,0], X[:,1], c = y, cmap = plt.cm.spring, edgecolor = 'k')
plt.show()

import numpy as np
clf = KNeighborsClassifier()
clf.fit(X, y)

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, .02),
                     np.arange(y_min, y_max, .02))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.pcolormesh(xx, yy, Z, cmap = plt.cm.Pastel1)
plt.scatter(X[:, 0], X[:, 1], c = y, cmap = plt.cm.spring, edgecolor = 'k')
plt.xlim(xx.min(), xx.max())
plt.ylim(xx.min(), yy.max())
plt.title("Classifier:KNN")
plt.show()

#打印模型评分
print('模型正确率：{:.2f}'.format(clf.score(X,y)))
