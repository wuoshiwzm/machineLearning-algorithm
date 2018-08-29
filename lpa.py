# 标签传递算法 LPA
import  random

def label_propagation(data, a):
    # 转移矩阵
    p = transition_matrix(data)
    m = len(data)#样本数目
    n = len(data[0])#样本维度
    for times in range(1000):#迭代1000次
        # i从a没有标记的一直到m-1,计算i号的样本，它的标签是谁，谁应该把标签传递给第i号
        for i in range(a, m):

            j = calc_label(p,i)
            label = data[j][n-1]#data[j]的标签
            if label>0:
                data[i][n-1] = label


def calc_label(p,i): #p:概率 pij:从顶点i转移到顶点j的概率
    n = len(p[i])
    k = random.random() #k belongsto [0,1)
    r = n-1
    for j in range(n):
        if p[i][j] >k: #如果Pij的转移概率大于随机数k，就把标签返回
            r=j
            break
    return r



# 转移矩阵 与laplace矩阵很像
def transition_matrix(data):
    m = len(data)
    p = [[] for x in range(m)]
    for i in range(m):
        p[i] = [0 for x in range(m)]
    nearest = [0 for x in range(neigbor)]

    for i in range(m):
        s = 0
        zero_list(nearest)
        for j in range(m):
            if i != j:
                p[i][j] = similar(data, i, j)
            else:
                p[i][j] = 0
            if is_neighbor(p[i][j], nearest):
                s += p[i][j]
            else:
                p[i][j] = 0
        for j in range(m):
            p[i][j] /= s
    return p
