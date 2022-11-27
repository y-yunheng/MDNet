import pandas as pd

data=pd.read_excel(r"D:\项目\Graduate_project\小论文\Datasets\Public_mental\train.xlsx")
a=[0]*6
for i in range(len(data)):
    a[eval(data["label"][i])[0]]+=1
print(a)