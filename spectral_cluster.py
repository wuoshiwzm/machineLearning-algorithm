# ---谱聚类Laplace算法（随机游走）

# 引用
import k_means


def spectral_cluster(data):
    # 生成laplace 矩阵
    lm = laplace_matrix(data)
    # 通过laplace矩阵 求特征值 特征向量
    eg_values, eg_vectors = linalg.eig(lm)
    # 排序
    idx = eg_values.argsort()
    eg_vectors = eg_vectors[: idx]

    m = len(data)
    # eg_data就是教程中的u
    eg_data = [[] for x in range(m)]
    for i in range(m)
        # u中的第i行 ：y[i]
        eg_data[i] = [0 for x in range(k)]

        # 选取前K个特征征
        for j in range(k)
            eg_data[i][j] = eg_vectors[i][j]
    return k_means(eg_data)


# laplace矩阵生成
def laplace_matrix(data):


    # m个样本
    m = len(data)
    w = [[] for x in range(m)]
    #初始化w矩阵
    for i in range(m):
        w[i] = [0 for x in range(m)]
    #最近
    nearest = [0 for x in range(neighbor)]

    for i in range(m):
        zero_list(nearest)
        for j in range(i+1,m):
            #计算第i和第j个相似度
            w[i][j] = similar(data,i,j)
            #如果不是前r个紧邻，就清零
            if not is_neighbor(w[i][j],nearest):
                w[i][j] = 0
            #否则就赋值，并对角化赋值
            w[j][i] = w[i][j]
        w[i][i] = 0
    for i in range(m):
        #求度s
        s=0
        for j in range(m):
            s+=w[i][j]
        if s==0:
            print('矩阵第',i,'行全为0')
            continue
        #随机游走的laplace矩阵
        for j in range(m):
            w[i][j] /= s
            w[i][j] = -w[i][j]

        #单位阵主对角线为1
        w[i][i] += 1
    return w


#算相似度
def is_similar(data, cluster_center):
    return

#
def nearest(data, cluster_center):
    return
