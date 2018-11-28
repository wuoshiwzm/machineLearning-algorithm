# k_means 算法

import random

print('''
in god we believe, 
others bring me the data
-leon''')


def k_means(data):
    # 给定k 为2
    k = 2
    # 长度
    m = len(data)
    # 维度
    n = len(data[0])
    # 所有样本尚未聚类 初始所有值设为-1 相当于都没做
    cluster = [-1 for x in range(m)]
    # 聚类中心  初始所有值都为[] 相当于都没有
    cluster_center = [[] for x in range(k)]
    # 下一轮的聚类中心
    cc = [[] for x in range(k)]
    # 每个簇中样本的数目
    c_number = [0 for x in range(k)]

    # 随机选择簇中心
    # 起始有0个簇中心 k为定好的k_means 要多少个
    i = 0
    # 做好k个聚类中心
    while i < k:

        j = random.randint(0, m - 1)
        # 如果和现有的聚类中心的值相似 就不选
        if is_similar(data[j], cluster_center):
            continue
        cluster_center[i] = data[j][:]
        # 把新的聚类中心赋为0
        cc[i] = [0 for x in range(n)]
        i += 1
    # 几千个样本 一般40次就够了
    for times in range(40):
        # 最多有m个样本
        for i in range(m):
            # 与第i个样本 最近的簇
            c = nearest(data[i], cluster_center)
            # 第i个样本归于第c簇
            cluster[i] = c
            c_number[c] += 1
            add(cc[c], data[i])
        for i in range(k):
            # cc[i] 除以c_number[i] 新的聚类中心
            divide(cc[i], c_number[i])
            c_number[i] = 0
            cluster_center[i] = cc[i][:]
            zero_list(cc[i])
        print(times, cluster_center)

    return cluster
