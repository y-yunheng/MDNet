import torch
from torch import nn


class Decoder(nn.Module):

    def __init__(self, input_dim, output_dim, hidden_dim, num_layers,
                 dropout,bidirectional=False,batch_first=False):  # 定义一个初始化函数，在Python类中规定，函数的第一个参数是实例对象本身，并且约定俗成，把其名字写为self。其作用相当于java中的this，表示当前类的对象，可以调用当前类中的属性和方法。
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, dropout=dropout,batch_first=batch_first,bidirectional=bidirectional)
        self.linner = nn.Linear(hidden_dim*2, output_dim)  # 输出层与输入层的维度相同

    def forward(self, input, hidden_h=None, hidden_c=None):  # 该神经网络向前传播函数
        if hidden_h==None and hidden_c==None:
            hidden_output, (hidden_h, hidden_c) = self.lstm(input)  # 这里的变量都是一维的因为我需要一个一个循环
        else:
            hidden_output, (hidden_h, hidden_c) = self.lstm(input, (hidden_h, hidden_c))  # 这里的变量都是一维的因为我需要一个一个循环
        outputs=torch.Tensor().cuda()
        for i in range(hidden_output.shape[1]):
            x=hidden_output[:,i,:]
            output = self.linner(x).unsqueeze(1)
            outputs=torch.cat((outputs,output),1)
          # 这里是在将tensor的输出转为一个常数
        return outputs, hidden_output, hidden_h, hidden_c