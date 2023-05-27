# import  numpy as np
# floderpath=r"D:\项目\Graduate_project\小论文\Log\model0\img\1confusion_matrix.np.npy"
# newnp=np.load(floderpath)
# print(newnp.sum(1))
# print(newnp.sum(0))
import numpy as np
import pandas as pd

modelname=["C-PsyD","FastText","TextCNN","ST-MFLC","BiLSTM","LSTM","Simple-RNN"]
print("Recall")
allrecall=[]
for k in range(0,7):
    floderpath = r"Log\model" + str(k) + r"\img\\1confusion_matrix.np.npy"
    confusion_matrix = np.load(floderpath)
    newnp = confusion_matrix.sum(0)
    newrecall=[0,0,0,0,0,0]
    for i in range(len(newnp)):
        if (newnp[i] != 0):
            newrecall[i] = round(confusion_matrix[i][i] / newnp[i],5)
    f = open(r"Log\model" + str(k) + "/img" + "/" + "recall.txt", "w")
    a = str(newrecall)
    newrecall=newrecall[1:]
    print(str(newrecall))
    allrecall.append(newrecall)
    f.write(a)

print("Far 假阳率  阴中预测为阳")
allfar=[]
for k in range(0,7):
    floderpath = r"Log\model" + str(k) + r"\img\\1confusion_matrix.np.npy"
    a = np.load(floderpath)
    allnum = np.sum(a, -1)
    allnum = np.sum(allnum, -1)
    newnp=[]
    for i in range(1, 6):
        alli = np.sum(a[i]) - a[i][i]
        wall = allnum - np.sum(a[:, i])
        newnp.append(round(alli / wall,5))
    f = open(r"Log\model" + str(k) + "/img" + "/" + "far.txt", "w")
    a = str(newnp)
    allfar.append(newnp)
    print(str(newnp))
    f.write(a)

print("准确率")
allacc=[]
for k in range(0,7):
    floderpath = r"Log\model" + str(k) + r"\img\\1confusion_matrix.np.npy"
    a = np.load(floderpath)
    allnum = np.sum(a, -1)
    allnum = np.sum(allnum, -1)
    newacc=[]
    TP=0
    for i in range(1, 6):
        TP+=a[i][i]
    newacc.append(round(TP/ allnum,5))
    f = open(r"Log\model" + str(k) + "/img" + "/" + "acc.txt", "w")
    a = str(newacc)
    print(str(newacc))
    allacc.append(newacc)
    f.write(a)

mdic={
    "模型":[],
    "准确率":[],
    "召回率0":[],
    "召回率1":[],
    "召回率2":[],
    "召回率3":[],
    "召回率4":[],
    "误报率0":[],
    "误报率1":[],
    "误报率2":[],
    "误报率3":[],
    "误报率4":[],
}

for  name,acc, far,recall in zip(modelname,allacc,allfar,allrecall):
    mdic["模型"].append(name)
    mdic["准确率"].append(str(round(acc[0]*100,1))+"%")
    mdic["误报率0"].append(str(round(far[0]*100,1))+"%")
    mdic["误报率1"].append(str(round(far[1]*100,1))+"%")
    mdic["误报率2"].append(str(round(far[2]*100,1))+"%")
    mdic["误报率3"].append(str(round(far[3]*100,1))+"%")
    mdic["误报率4"].append(str(round(far[4]*100,1))+"%")
    mdic["召回率0"].append(str(round(recall[0]*100,1))+"%")
    mdic["召回率1"].append(str(round(recall[1]*100,1))+"%")
    mdic["召回率2"].append(str(round(recall[2]*100,1))+"%")
    mdic["召回率3"].append(str(round(recall[3]*100,1))+"%")
    mdic["召回率4"].append(str(round(recall[4]*100,1))+"%")

df2 = pd.DataFrame(mdic)

        # 将 DataFrame 写入 CSV 文件
df2.to_excel("结果.xlsx", index=False)
