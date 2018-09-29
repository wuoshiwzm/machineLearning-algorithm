# Viterbi算法
def viterbi(pi, A, B, o):
    T = len(o)  # 观测序列
    delta = [[0 for i in range(4)] for t in range(T)]
    pre = [[0 for i in range(4)] for t in range(T)]  # 前一个状态

    for i in range(4):
        delta[0][i] = pi[i] + B[i][ord(o[0])]
    for t in range(1, T):
        for i in range(4):
            delta[t][i] = delta[t - 1][0] + A[0][i]
            for j in range(1, 4):
                vj = delta[t - 1][j] + A[j][i]
                if delta[t][i] < vj:
                    delta[t][i] = vj
                    pre[t][i] = j
            delta[t][i] += B[i][ord(o[t])]
    decode = [-1 for t in range(T)]  # 解码：回溯查找概率最大路径

    q = 0
    for i in range(1, 4):
        if delta[T-1][i]>delta[T-1][q]:
            q= i
    decode[T-1] = q

    for t in range(T-2,-1,-1):
        q = pre[t+1][q]
        decode[t] = q
    return decode
