# 分词  HMM应用
def segment(sentence, decode):
    N = len(sentence)
    i = 0
    while i < N:
        if decode[i] == 0 or decode[i] == 1:  # Begin
            j = i + 1
            while j < N:
                if decode[j] == 2:
                    break
                j += 1
            print(sentence[i:j+1], "/")
            i = j+1
        elif decode[i] == 3 or decode[i] == 2: #single
            print(sentence[i:i+1],"/")
            i+=1
        else:
            print('error:',i,decode[i])
            i += 1
