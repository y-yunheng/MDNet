import random

import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import pandas as pd


class MyDatasetPandas(Dataset):
    """
        下载数据、初始化数据，都可以在这里完成
        本类不对数据进行处理，仅仅返回语句与标签
    """

    def __init__(self, flie,maxlen=200):
        """
        步骤二：实现构造函数，定义数据读取方式，划分训练和测试数据集
        """
        super(MyDatasetPandas, self).__init__()
        self.df = pd.read_excel(flie)
        self.maxlen=maxlen
        self.len = len(self.df)

    def __getitem__(self, index):
        """
        步骤三：实现__getitem__方法，定义指定index时如何获取数据，并返回单条数据（训练数据，对应的标签）
        """
        words=self.df["title"][index]
        label = eval(self.df["label"][index])[0]
        return words,np.array([label],dtype="int64")
    def __len__(self):
        """
        步骤四：实现__len__方法，返回数据集总数目
        """
        return self.len

    def insert_noise(self,words,num):
        new_words=[]
        words=list(words)
        noise=['，','。',' ',' ?']
        for i in range(num):
            inerst_id = random.randint(0, len(noise))
            v = noise[inerst_id]
            words.insert(random.randint(0, len(words)), v)

        return str(words)


















#利用Pands加载

