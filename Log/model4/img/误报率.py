import numpy as np

a=np.load(r"D:\项目\小论文\MDNet\Log\model4\img\1confusion_matrix.np.npy")
print(a)
allnum=np.sum(a,-1)
allnum=np.sum(allnum,-1)
for i in range(1,6):
    N = allnum - np.sum(a[:, i])
    P = np.sum(a[:, i])

    TP=a[i][i]
    FP=np.sum(a[i])-a[i][i]
    FN=N-FP


    #print("第",i,"类的误报率为：", TP/N)
    print("第",i,"类的准确率为：", (TP+FN)/(P+N))


