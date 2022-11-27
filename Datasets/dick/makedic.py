import  numpy as np
import json
datakw={}
datawk={}
dickw=np.load("D:\项目\Matser_graduation_project\TextClass\PublicTool\dick\字向量kw.npy",allow_pickle=True)
dicwk = np.load("D:\项目\Matser_graduation_project\TextClass\PublicTool\dick\字向量wk.npy",allow_pickle=True)
i=0
for values in dicwk.item():
    datakw[i] = values
    datawk[values] = i
    i+=1

js = str(dicwk)
file = open('word_wk.txt', 'w')
file.write(js)
file.close()


js = str(datakw)
file = open('word_kw.txt', 'w')
file.write(js)
file.close()