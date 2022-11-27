import numpy as np

a=np.load(r"D:\项目\Graduate_project\小论文\Log\model1\img\1confusion_matrix.np.npy")
print(a)
allnum=np.sum(a,-1)
allnum=np.sum(allnum,-1)
for i in range(1,6):
    alli=np.sum(a[i])-a[i][i]
    wall=allnum-np.sum(a[:,i])
    print("第",i,"类的误报率为：", alli/wall)
