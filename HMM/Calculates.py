#计算相关函数，前向-后向 alpha(t,i) beta(t,j)
def calc_alpha(pi,A,B,o,alpha):
    for i in range(4):
        alpha[0][i] = pi[i]+B[i][ord(o[0])]
    T = len(o)
    temp = [0 for i in range(4)]
    del i
    for t in range(1,T):
        for i in range(4):
            for j in range(4):
                temp[j]=(alpha[t-1][j] + A[j][i])
            alpha[t][i] = log_sum(temp)
            alpha[t][i] +=B[i][ord(o[t])]

def calc_beta(pi,A,B,o,beta):
    T= len(o)
    for i in range(4):
        beta[T-1][i] = 1
    temp = [0 for i in range(4)]
    del i#删除i
    for t in range(T-2,-1,-1):
        for i in range(4):
            #ord()函数是chr()或unichr()的配对函数，以字符作为参数，
            # 返回ASCII数值，或者Unicode数值。o[t+1]:t+1时刻的观测值
            temp[j] = A[i][j] + B[j][ord(o[t+1])] + beta[t+1][j]
        beta[t][j] += log_sum(temp)


# 隐状态概率 - 隐状态转移概率
def calc_gamma(alpha,beta,gamma):
    for t in range(len(alpha)):
        for i in range(4):
            gamma[t][i] = alpha[t][i] + beta[t][i]
        s = log_sum(gamma[t])
        for i in range(4):
            gamma[t][i] -= s

def calc_ksi(alpha,beta,A,B,o,ksi):
    T = len(alpha)
    temp = [0 for x in range(16)]
    for t in range(T-1):
        k=0
        for i in range(4):
            for j in range(4):
                ksi[t][i][j] = alpha[t][i] + A[i][j] +B[j][ord(o[t+1])] + beta[t+1][j]
                temp[k] = ksi[t][i][j]
                k+=1
        s=log_sum(temp)#把所有的加起来，得到分母
        for i in range(4):
            for j in range(4):
                ksi[t][i][j] -=s

