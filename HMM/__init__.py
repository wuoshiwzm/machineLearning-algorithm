if __name__ == "__main__":#初始化pi,A,B
    pi=[random.random() for x in range(4)] #初始分布  pi
    log_normalize(pi)

    A=[[random.random() for y in range(4)] for x in range(4)]#转移矩阵 4*4
    A[0][0] = A[0][3] = A[1][0] = A[1][3]\
            =A[2][1] = A[2][2] = A[3][1] \
        = A[3][2] = 0  #不可能事件
    B = [[random.random() for y in range(65535)] for x in range(4)]

    for i in range(4):
        log_normalize(A[i])
        log_normalize(B[i])
    baum_welch(pi,A,B)
    save_parameter(pi,A,B)