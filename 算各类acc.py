# import  numpy as np
# floderpath=r"D:\项目\Graduate_project\小论文\Log\model0\img\1confusion_matrix.np.npy"
# newnp=np.load(floderpath)
# print(newnp.sum(1))
# print(newnp.sum(0))
import numpy as np

print("Recall")
for k in range(0,7):
    floderpath = r"D:\项目\小论文\MDNet\Log\model" + str(k) + r"\img\\1confusion_matrix.np.npy"
    confusion_matrix = np.load(floderpath)
    newnp = confusion_matrix.sum(0)
    newrecall=[0,0,0,0,0,0]
    for i in range(len(newnp)):
        if (newnp[i] != 0):
            newrecall[i] = confusion_matrix[i][i] / newnp[i]
    f = open(r"D:\项目\小论文\MDNet\Log\model" + str(k) + "/img" + "/" + "recall.txt", "w")
    a = str(newrecall)
    print(str(newrecall))
    f.write(a)

print("Far 假阳率  阴中预测为阳")
for k in range(0,7):
    floderpath = r"D:\项目\小论文\MDNet\Log\model" + str(k) + r"\img\\1confusion_matrix.np.npy"
    a = np.load(floderpath)
    allnum = np.sum(a, -1)
    allnum = np.sum(allnum, -1)
    newnp=[]
    for i in range(1, 6):
        alli = np.sum(a[i]) - a[i][i]
        wall = allnum - np.sum(a[:, i])
        newnp.append(alli / wall)
    f = open(r"D:\项目\小论文\MDNet\Log\model" + str(k) + "/img" + "/" + "far.txt", "w")
    a = str(newnp)
    print(str(newnp))
    f.write(a)

print("准确率")
for k in range(0,7):
    floderpath = r"D:\项目\小论文\MDNet\Log\model" + str(k) + r"\img\\1confusion_matrix.np.npy"
    a = np.load(floderpath)
    allnum = np.sum(a, -1)
    allnum = np.sum(allnum, -1)
    newnp=[]
    for i in range(1, 6):
        TP=a[i][i]
        FN=allnum-np.sum(a[:,i])-np.sum(a[i])+a[i][i]

        newnp.append((TP+FN) / allnum)
    f = open(r"D:\项目\小论文\MDNet\Log\model" + str(k) + "/img" + "/" + "acc.txt", "w")
    a = str(newnp)
    print(str(newnp))
    f.write(a)