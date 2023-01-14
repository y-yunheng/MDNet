#这是一个文本分类框架，该框架采用Seq2Seq
import time
import pynvml

from Tool.Accuracy import MutiAccuracy
from Tool.DrawImag.DrawImag import Draw_Confusion_matrix
from Tool.MyDataset import MyDatasetPandas
from Tool.MyLoss import CrossEntropyCriterion
from model.model import FastText, MDnet, BiLSTM, ST_MFLC, TextCNN

pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)
meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
import pandas
import pandas as pd
import torch
from pandas import DataFrame
from torch.utils.data import DataLoader


DEVICE=torch.device('cuda'if torch.cuda.is_available()else'cpu')

modelname=["model0","model1","model2","model3","model4"]
models=[ MDnet(), FastText(), TextCNN(),ST_MFLC(), BiLSTM()]
modelindex=4
try:
    #尝试加载模型
    model =torch.load("Log/"+modelname[modelindex]+"/work/model.pt")
    print("加载模型成功")
except:
    #加载模型失败新建模型
    print("加载模型失败")
    model =models[modelindex]


#设置相关数据集，设置好相关训练参数，包括数据加载器，训练epoch数

#设置训练次数、批次
Epoch=5
Batch_Szie=64
train_dataset = MyDatasetPandas(r"Datasets\Public_mental\train.xlsx")
test_dataset = MyDatasetPandas(r"Datasets\Public_mental\test.xlsx")
train_loader = DataLoader(train_dataset, batch_size=Batch_Szie, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
Data=[Epoch,train_loader,test_loader]
#示例化模型训练类
loss_fn=CrossEntropyCriterion()#声明训练过程中使用的loss函数
#设置训练记录日志
mdic={}
mdic["epoch"]=[]
mdic["batch_id"] =[]
mdic["loss"] =[]
mdic["acc1"] = []
mdic["acc2"] = []
mdic["acc3"] = []
mdic["time_consuming"]=[]
mdic["gpu_use"]=[]
rt_test={}
rt_test["epoch"]=[]
rt_test["acc1"] = []
#开始训练
acc_fn=MutiAccuracy()#声明准确度函数
model.train()
model.to(DEVICE)
optimer=torch.optim.Adam(params=model.parameters())
# 梯度清零
optimer.zero_grad()
allstep = 0

for epoch in range(Epoch):
    model.train()
    starttime=time.time()
    acc_fn.reset()
    for batchid, data in enumerate(train_loader):
        # 训练数据
        words = list(data[0])
        labels = data[1]
        predicts = model(words)  # 预测结果
        # 计算损失 等价于 prepare 中loss的设置
        loss = loss_fn(predicts.to(DEVICE), labels.to(DEVICE))
        # 计算准确率 等价于 prepare 中metrics的设置
        acc = acc_fn.getacc(predicts, labels)
        # 反向传播
        loss.backward()
        # 更新参数
        optimer.step()
        # 梯度清零
        optimer.zero_grad()
        allstep += 1
        # torch.cuda.empty_cache()
        mdic["epoch"].append(epoch)
        mdic["batch_id"].append(batchid)
        mdic["loss"].append(float(loss.detach().cpu().numpy()))
        mdic["acc1"].append(float(acc[0]))
        mdic["time_consuming"].append(time.time() - starttime)
        mdic["gpu_use"].append(meminfo.used / (1024 * 1024))
        if (batchid + 1) % 10 == 0:
            print(
                "epoch: {}, batch_id: {}, loss is: {}, acc is: {}".format(epoch, batchid ,loss.detach().cpu().numpy(), acc))
            #torch.save(model, "Log/"+modelname[0]+"/work/model.pt")


    model.eval()
    acc_fn.reset()
    for batchid, data in enumerate(test_loader):
        # 训练数据
        words = list(data[0])
        labels = data[1]
        predicts = model(words)  # 预测结果
        # 计算损失 等价于 prepare 中loss的设置
        loss = loss_fn(predicts.to(DEVICE), labels.to(DEVICE))
        # 计算准确率 等价于 prepare 中metrics的设置
        acctest = acc_fn.getacc(predicts, labels)
        allstep += 1
        # torch.cuda.empty_cache()
    print("epoch: {},  acc is: {}".format(epoch,acctest))
    rt_test["epoch"].append(epoch)
    rt_test["acc1"].append(acctest)
    dataframe = pd.DataFrame(rt_test, columns=["epoch", "acc1"])
    dataframe.to_excel("Log/" + str(modelname[modelindex]) + "/rt_test_log.xlsx")

dataframe = pd.DataFrame(mdic,columns=["epoch","batch_id","loss","acc1","time_consuming","gpu_use"])
dataframe.to_excel("Log/"+str(modelname[modelindex])+"/train_log.xlsx")


#开始测试
acc_fn=MutiAccuracy()
dc=Draw_Confusion_matrix([6],1)
testdic={}
testdic["epoch"]=[]
testdic["batch_id"] =[]
testdic["loss"] =[]
testdic["acc1"] = []
testdic["acc2"] = []
testdic["acc3"] = []
testdic["time_consuming"]=[]
testdic["gpu_use"]=[]
print("开始测试")
model.eval()
model.to(DEVICE)
optimer=torch.optim.Adam(params=model.parameters())
# 梯度清零
optimer.zero_grad()
allstep = 0
for batchid, data in enumerate(test_loader):
    # 训练数据
    words = list(data[0])
    labels = data[1]
    predicts = model(words)  # 预测结果
    # 计算损失 等价于 prepare 中loss的设置
    loss = loss_fn(predicts.to(DEVICE), labels.to(DEVICE))
    # 计算准确率 等价于 prepare 中metrics的设置
    acc = acc_fn.getacc(predicts, labels)
    allstep += 1
    # torch.cuda.empty_cache()
    testdic["epoch"].append(0)
    testdic["batch_id"].append(batchid)
    testdic["loss"].append(float(loss.detach().cpu().numpy()))
    testdic["acc1"].append(float(acc[0]))

    dc.compute_matrix(predicts,labels)
    if (batchid + 1) % 1 == 0:
        print(
            "测试: {}, batch_id: {}, loss is: {}, acc is: {}".format(0, batchid, loss.detach().cpu().numpy(), acc))

dc.save_matrix("Log/"+modelname[modelindex]+"/img")
dataframe = pd.DataFrame(testdic,columns=["epoch","batch_id","loss","acc1"])
dataframe.to_excel("Log/"+modelname[modelindex]+"/test_log.xlsx")