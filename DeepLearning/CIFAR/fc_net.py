# 两层的神经网络
from layer_utils import *
import numpy as np


class TwoLayerNet(object):
    """
    初始化，图片数据像素为32x32, 有3个色域，所以是3x32x32
    input_dim: 输入的维度
    hidden_dim:隐层的 神经元个数
    num_classes:分类的数量
    weight_scale:进行权重初始化时，不希望太大，希望比较小，就要乘上这个值，这里是1e-3 (0.001)
    reg:L2正则化时，惩罚权重项  (1/2 * W^2)
    """

    def __init__(self, input_dim=3 * 32 * 32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.01):
        # 对参数w1 w2  b1 b2进行初始化
        self.params = {}
        self.reg = reg
        # w1,b1, 为输入到隐层，W1为ax100维

        self.params['W1'] = weight_scale * np.random.randn(input_dim, hidden_dim)
        self.params['b1'] = np.zeros((1, hidden_dim))
        # w2,b2, 为隐层到输出层，输出为10维的(10分类)，W1为ax100维
        self.params['W2'] = weight_scale * np.random.randn(input_dim, hidden_dim)
        self.params['b2'] = np.zeros((1, hidden_dim))
