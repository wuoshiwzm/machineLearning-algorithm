def bw(pi,A,B,alpha,beta,gamma,ksi,o):
    T = len(alpha)

    #求A矩阵
    for i in range(4):
        pi[i]=gamma[0][i]
    s1 = [0 for x in range(T-1)]
    s2 = [0 for x in range(T-1)]
    for i in range(4):
        for j in range(4):
            s1[t] = ksi[t][i][j]#《统计学习方法》(10.39)
            s2[t] = gamma[t][i]#《统计学习方法》(10.40)
        A[i][j] = log_sum(s1) - log_sum(s2)
    s1 = [0 for x in range(T)]
    s2 = [0 for x in range(T)]

    #求B矩阵
    for i in range(4):
        for k in range(65535):
            valid = 0
            for t in range(T):
                if ord(o[t]) == k:#判断o(t) = v(k)
                    s1[valid] = gamma[t][i]
                    valid += 1
                s2[t] = gamma[t][i]
            if valid == 0:
                B[i][k] = infinite
            else:
                B[i][k] = log_sum(s1[:valid]) - log_sum(s2)