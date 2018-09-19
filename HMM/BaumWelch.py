#Baum-Welch算法
def baum_welch(pi,A,B):
    f = file(".\\text\\1.txt")
    sentence = f.read()[3:].decode('utf-8')#跳过文件头
    f.close()
    T = len(sentence)   #观测序列

    alpha = [[0 for i in range(4)] for t in range(T)]
    beta = [[0 for i in range(4)] for t in range(T)]
    gamma = [[0 for i in range(4)] for t in range(T)]
    ksi = [[[0 for j in range(4)] for i in range(4)]for t in range(T-1)]

    for time in range(100):

        calc_alpha(pi,A,B,sentence,alpha) #前向概率alpha(t,i):给定lamda,
        # 在时候t部分观测序列为o1,o2,...,ot,且状态为i的概率

        calc_beta(pi,A,B,sentence,beta)   #后向概率beta(t,i):给定lamda和时刻t的状态i,
        # 从t+1到T的部分观测序列为ot+1,ot+2,...,oT的概率

        calc_gamma(alpha,beta,gamma)      #gamma(t,i):给定lamda和O，在时刻t状态为i的概率

        calc_ksi(alpha,beta,A,B,sentence,ksi) #ksi(t,i,j):给定lamda和O,
        # 在时刻t状态为i,在时刻t+1时处于状态j的概率

        bw(pi,A,B,alpha,beta,gamma,ksi,sentence)#baum_welch算法
