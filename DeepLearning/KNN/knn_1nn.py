# k近邻算法(k=1的情况)
import numpy as np


class NearestNeighbor:
    def __init__(self):
        pass

    def train(self, X, y):
        # 对k近邻算法，没有模型学习过程，分类器就是所有学习数据本身
        self.Xtr = X
        self.ytr = y

    def predict(self, X):
        # X is N*D where each row is an example we wish to predict label for
        # num_test 数量就是输入的条数
        num_test = X.shape[0]
        # make sure that the output type matches the input type
        Ypred = np.zeros(num_test, dtype=self.ytr.dtype)
        for i in xrange(num_test):
            # 计算 L1 距离，find the nearest training image tto the i'th test image
            # L1:sum of absolute value differences
            # axis : None or int or tuple of ints, optional
            # 平时用的sum应该是默认的axis=0 就是普通的相加
            # 而当加入axis=1以后就是将一个矩阵的每一行向量相加,即每一条数据对应的L1距离求和
            distance = np.sum(np.abs(self.Xtr - X[i, :]), axis=1)
            min_index = np.argmin(distance)  # 求最近邻，注意这里是1近邻
            Ypred[i] = self.ytr[min_index]  # 把距离最近的点做为测试数据的类
        return Ypred
