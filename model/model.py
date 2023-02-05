import json

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence
import torch.nn.functional as F
class Self_Attention(nn.Module):

    # 用来实现mask-attention layer
    def __init__(self, hidden_dim, is_bi_rnn):
        super(Self_Attention, self).__init__()

        self.hidden_dim = hidden_dim
        self.is_bi_rnn = is_bi_rnn

        # 下面使用nn的Linear层来定义Q，K，V矩阵
        if is_bi_rnn:
            # 是双向的RNN
            self.Q_linear = nn.Linear(hidden_dim * 2, hidden_dim * 2, bias=False)
            self.K_linear = nn.Linear(hidden_dim * 2, hidden_dim * 2, bias=False)
            self.V_linear = nn.Linear(hidden_dim * 2, hidden_dim * 2, bias=False)
        else:
            # 单向的RNN
            self.Q_linear = nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.K_linear = nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.V_linear = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, inputs, lens):

        size = inputs.size()
        # 计算生成QKV矩阵
        #此处将查询张量定义为inputs
        Q = self.Q_linear(inputs)
        K = self.K_linear(inputs)  # 先进行一次转置
        V = self.V_linear(inputs)

        #开始计算权重
        W=F.softmax(torch.matmul(Q,K.transpose(1,2))/pow(self.hidden_dim,0.5))
        maxlen=max(lens)
        mask=[]
        for mlen in lens:
            a=[1]*mlen+[0]*(maxlen-mlen)
            mask.append(a)
        mask=torch.tensor(mask).unsqueeze(-1).cuda()
        mask=mask.expand(W.shape)
        W=torch.mul(W,mask)
        Z= torch.bmm(W,V)
        res=torch.sum(Z,1).unsqueeze(1)
        return res
class Attention(nn.Module):
    def __init__(self,hidden_size,bidirectional=False):  # 定义一个初始化函数，在Python类中规定，函数的第一个参数是实例对象本身，并且约定俗成，把其名字写为self。其作用相当于java中的this，表示当前类的对象，可以调用当前类中的属性和方法。
        super().__init__()
        if(bidirectional):
            self.hidden_size = hidden_size*2
        else:
            self.hidden_size = hidden_size
        self.Wa = nn.Linear(self.hidden_size*2, self.hidden_size)
        self.VT = nn.Linear(self.hidden_size, 1)

    def forward(self, s, encoder_hidden_output,lens):
        #s值是用来与encoder_hidden_output计算权值的
        res=self.score(s,encoder_hidden_output)
        maxlen = max(lens)
        mask = []
        for mlen in lens:
            a = [1] * mlen + [0] * (maxlen - mlen)
            mask.append(a)
        mask = torch.tensor(mask).cuda()

        attention_weight=torch.mul(res, mask)
        attention_weight = F.softmax(attention_weight, dim=-1)
        # attention_weight [batch_size,step_time]
        attention_res=encoder_hidden_output.multiply(attention_weight.unsqueeze(-1))
        #  attention_res[batch_size,step_time,hidden_size]
        attention_sumVector=torch.sum(attention_res, dim=1).unsqueeze(1)
        # attention_sumVector[batch_size,1,hidden_size]
        return attention_sumVector
    def score(self, s, h):
        # 此函数为相似度计算 计算某时刻解码器的隐层向量与编码器所有隐层向量的相似度。并以此为权值
        # 关键函数:输入S与H 从而得到e，进而得出权值，以c为s0，h1为s1 h2为s2....，此为相关性计算
        # 输出：权值

        s = s.expand(h.shape)
        # s[batch_size,step_time,hidden_size]
        x=torch.concat([s, h], dim=-1)
        sh=self.Wa(x)
        #过一遍激活函数
        sh=torch.tanh(sh)
        #继续过神经网络
        res = self.VT(sh)
        # res[batch_size,step_time,1]
        res = res.squeeze(-1)
        # res[batch_size,step_time]
        return res


class MDnet(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=128, num_layers=2, output_dim=6, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, 2, dropout=dropout, bidirectional=True, batch_first=True)


        self.dickw = self.readidc("Datasets\dick\word_kw.txt")
        self.dicwk = self.readidc("Datasets\dick\word_wk.txt")
        self.embedder = nn.Embedding(20000, input_dim)
        self.attention = Attention(hidden_dim, bidirectional=True)
        self.self_attention = Self_Attention(hidden_dim,True)

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=2, stride=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=2, stride=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=2, stride=1),
            nn.ReLU(),
        )

        # self.dropoutconv=nn.Dropout(dropout)
        self.dropoutrnn = nn.Dropout(dropout)
        # self.dropoutsa = nn.Dropout(dropout)
        #
        # self.lstm_grufc=nn.Linear(hidden_dim * 2 , hidden_dim )
        # self.convfc=nn.Linear(3990,256)
        # self.cnnfc = nn.Linear(hidden_dim * 2, output_dim)
        self.rnnfc = nn.Linear(hidden_dim * 2, output_dim)  # 输出层与输入层的维度相同
        # self.safc = nn.Linear(hidden_dim* 2 , output_dim)  # 输出层与输入层的维度相同
        # self.outfc = nn.Linear(output_dim*2 , output_dim)  # 输出层与输入层的维度相同
    def forward(self, inputs):
        tensorwords, seqlen = self.convert_woed2ids(inputs)
        embed_tensor = self.convert_ids2vector(tensorwords.cuda())
        data = pack_padded_sequence(embed_tensor, seqlen, batch_first=True, enforce_sorted=False).cuda()
        pad_data, seq_len = pad_packed_sequence(data, batch_first=True)


        conv_input_data = embed_tensor  # , seq_len = pad_packed_sequence(data, batch_first=True)
        conv_input_data = conv_input_data.transpose(1, 2)
        conv_out = self.conv1(conv_input_data).transpose(1, 2)
        conv_out = pack_padded_sequence(conv_out, seqlen, batch_first=True, enforce_sorted=False).cuda()
        conv_out_data,_=pad_packed_sequence(conv_out, batch_first=True)

        # 开始自注意力运算
        #self_attention_output = self.self_attention(pad_data, seq_len)
        # self_fc_out = self.safc(torch.mean(self_attention_output, 1).unsqueeze(1))

        # lstm_out, _ = self.lstm(conv_out_data)
        # lstm_out_data,seq_len=pad_packed_sequence(lstm_out, batch_first=True)
        # gru_input_data=self.lstm_grufc(lstm_out_data)
        #gru_input=pack_padded_sequence(self_attention_output, seqlen, batch_first=True, enforce_sorted=False).cuda()
        encoder_out_gru, _ = self.gru( data )
        encoder_hidden_output_gru, seq_len = pad_packed_sequence(encoder_out_gru, batch_first=True)

        # encoder_out_lstm, _ = self.lstm(data)
        # encoder_hidden_output_lstm, seq_len = pad_packed_sequence(encoder_out_gru, batch_first=True)

        #conv_out_data = pack_padded_sequence(self_attention_output, seqlen, batch_first=True,enforce_sorted=False).cuda()
        # 注意力机制运算
        t=torch.mean(conv_out_data, 1).unsqueeze(-1)
        t=t.expand(t.shape[0],t.shape[1],t.shape[2]*2)
        t=t.reshape([t.shape[0],1,256])
        s = t#self.get_s(self_attention_output, seq_len)

        self_attention_output = self.self_attention(torch.cat([pad_data,pad_data],-1), seq_len)


        dinput1 = self.attention(s, encoder_hidden_output_gru, seq_len)
        dinput2= self.attention(self_attention_output, encoder_hidden_output_gru, seq_len)
        dinput=torch.mean(torch.cat([dinput1,dinput2],1),1).unsqueeze(1)
        x=self.dropoutrnn(dinput)
        gru_output=self.rnnfc(x)


        # x = self.dropoutsa(self_attention_output)
        # sa_output = self.safc(x)
        #
        # output=self.outfc(torch.cat([gru_output,sa_output],-1))

        return gru_output

    def get_s(self, encoder_hidden_output, seqlen):
        batch = []
        for i in range(len(seqlen)):
            batch.append(i)
        index = seqlen - 1
        s = encoder_hidden_output[batch, index, :]
        return s.unsqueeze(1)

    def convert_woed2ids(self, words):
        listids = []

        seqlen = []
        for oneword in words:
            listids_one = []
            for char in oneword:
                if char not in self.dicwk.keys():
                    listids_one.append(self.dicwk['puk'])
                else:
                    listids_one.append(self.dicwk[char])
            seqlen.append(len(listids_one))
            while (len(listids_one) < 4000):
                listids_one.append(self.dicwk['pad'])
            listids.append(listids_one)
        return torch.tensor(listids), seqlen

    def convert_ids2vector(self, tensorwords):

        return self.embedder(tensorwords)

    def readidc(self, path):
        file = open(path, 'r', encoding="utf-8")
        js = file.read()
        dic = eval(js)
        file.close()
        return dic


class FastText(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=128, num_layers=2, output_dim=6, dropout=0.3):
        super().__init__()
        self.dickw = self.readidc("Datasets\dick\word_kw.txt")
        self.dicwk = self.readidc("Datasets\dick\word_wk.txt")
        self.embedder = nn.Embedding(20000, input_dim)

        self.fc = nn.Linear(128, output_dim)  # 输出层与输入层的维度相同

    def forward(self, inputs):
        tensorwords, seqlen = self.convert_woed2ids(inputs)
        embed_tensor = self.convert_ids2vector(tensorwords.cuda())
        FastText_out = torch.mean(embed_tensor, 1)
        FastText_out=FastText_out.unsqueeze(1)

        output=self.fc(FastText_out)

        return output

    def convert_woed2ids(self,words):
        listids=[]

        seqlen=[]
        for oneword in  words:
            listids_one = []
            for char in oneword:
                if char not in self.dicwk.keys():
                    listids_one.append(self.dicwk['puk'])
                else:
                    listids_one.append(self.dicwk[char])
            seqlen.append(len(listids_one))
            while(len(listids_one)<4000):
                listids_one.append(self.dicwk['pad'])
            listids.append(listids_one)
        return torch.tensor(listids),seqlen
    def convert_ids2vector(self,tensorwords):

        return self.embedder(tensorwords)
    def readidc(self,path):
        file = open(path, 'r',encoding="utf-8")
        js = file.read()
        dic = eval(js)
        file.close()
        return dic

class TextCNN(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=128, num_layers=2, output_dim=6, dropout=0.3):
        super().__init__()
        self.dickw = self.readidc("Datasets\dick\word_kw.txt")
        self.dicwk = self.readidc("Datasets\dick\word_wk.txt")
        self.embedder = nn.Embedding(20000, input_dim)

        self.conv= nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=32, kernel_size=2, stride=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=32, out_channels=1, kernel_size=2, stride=1),
            nn.ReLU(),
        )
        self.maxpool=nn.MaxPool1d(2)
        self.avgpool = nn.AvgPool1d(2)
        self.fn=nn.Flatten()
        self.fc = nn.Linear(3998, output_dim)  # 输出层与输入层的维度相同

    def forward(self, inputs):
        tensorwords, seqlen = self.convert_woed2ids(inputs)
        embed_tensor = self.convert_ids2vector(tensorwords.cuda())
        conv_out=self.conv( embed_tensor.transpose(1,2))
        maxout=self.maxpool(conv_out)
        avgout = self.avgpool(conv_out)
        cnn_out=torch.cat([maxout,avgout],-1).squeeze(1)
        cnn_fc_out=self.fn(cnn_out)
        output=self.fc(cnn_fc_out)

        return output.unsqueeze(1)

    def convert_woed2ids(self,words):
        listids=[]

        seqlen=[]
        for oneword in  words:
            listids_one = []
            for char in oneword:
                if char not in self.dicwk.keys():
                    listids_one.append(self.dicwk['puk'])
                else:
                    listids_one.append(self.dicwk[char])
            seqlen.append(len(listids_one))
            while(len(listids_one)<4000):
                listids_one.append(self.dicwk['pad'])
            listids.append(listids_one)
        return torch.tensor(listids),seqlen
    def convert_ids2vector(self,tensorwords):

        return self.embedder(tensorwords)
    def readidc(self,path):
        file = open(path, 'r',encoding="utf-8")
        js = file.read()
        dic = eval(js)
        file.close()
        return dic

class ST_MFLC(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=128, num_layers=2, output_dim=6, dropout=0.2):
        super().__init__()
        self.dickw = self.readidc("Datasets\dick\word_kw.txt")
        self.dicwk = self.readidc("Datasets\dick\word_wk.txt")
        self.cnn_embed=nn.Embedding(20000, input_dim)
        self.TextCNN = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=64, kernel_size=2, stride=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=32, kernel_size=2, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(in_channels=32, out_channels=16, kernel_size=2, stride=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=16, out_channels=1, kernel_size=2, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(2),

        )
        self.fccnn = nn.Linear(998, output_dim)  # 输出层与输入层的维度相同


        self.TextRNN = nn.Sequential(
            nn.Embedding(20000, input_dim),
            nn.LSTM(input_dim, hidden_dim, num_layers, dropout=dropout, bidirectional=True, batch_first=True),
        )
        self.fcRNN = nn.Linear(256, 6)

        self.embed_sa = nn.Embedding(20000, input_dim)
        self.self_attention=Self_Attention(hidden_dim, False)
        self.fcsa = nn.Linear(128, output_dim)  # 输出层与输入层的维度相同


        self.linear = nn.Linear(output_dim * 3, output_dim)  # 输出层与输入层的维度相同
        self.sg=nn.Sigmoid()

    def forward(self, inputs):
        tensorwords, seqlen = self.convert_woed2ids(inputs)
        data = pack_padded_sequence(tensorwords, seqlen, batch_first=True, enforce_sorted=False).cuda()
        input_data,seq_len = pad_packed_sequence(data, batch_first=True)

        cnndata = pack_padded_sequence(tensorwords, [4000]*input_data.shape[0], batch_first=True, enforce_sorted=False).cuda()
        cnn_input_data, cnn_seq_len = pad_packed_sequence(cnndata, batch_first=True)

        cnn_input_data_embed=self.cnn_embed(cnn_input_data)
        cnnout=self.TextCNN(cnn_input_data_embed.transpose(1,2))
        cnnout=self.fccnn(cnnout)


        rnnout,_ = self.TextRNN(input_data)
        rnnout=rnnout[:, -1, :]
        rnnout=self.fcRNN(rnnout).unsqueeze(1)

        input_embed_sa=self.embed_sa(input_data)
        saout = self.self_attention(input_embed_sa,seq_len)
        saout=self.fcsa(saout)

        allout=torch.cat([cnnout,rnnout,saout],-1)
        allout=self.linear(allout)
        allout=self.sg(allout)



        return allout

    def get_s(self, encoder_hidden_output, seqlen):
        batch = []
        for i in range(len(seqlen)):
            batch.append(i)
        index = seqlen - 1
        s = encoder_hidden_output[batch, index, :]
        return s.unsqueeze(1)

    def convert_woed2ids(self, words):
        listids = []

        seqlen = []
        for oneword in words:
            listids_one = []
            for char in oneword:
                if char not in self.dicwk.keys():
                    listids_one.append(self.dicwk['puk'])
                else:
                    listids_one.append(self.dicwk[char])
            seqlen.append(len(listids_one))
            while (len(listids_one) < 4000):
                listids_one.append(self.dicwk['pad'])
            listids.append(listids_one)
        return torch.tensor(listids), seqlen


    def readidc(self, path):
        file = open(path, 'r', encoding="utf-8")
        js = file.read()
        dic = eval(js)
        file.close()
        return dic



class CNN3_LSTM(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=128, num_layers=2, output_dim=6, dropout=0.2):
        super().__init__()
        self.dickw = self.readidc("Datasets\dick\word_kw.txt")
        self.dicwk = self.readidc("Datasets\dick\word_wk.txt")
        self.embedder = nn.Embedding(20000, input_dim)

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(in_channels=32, out_channels=1, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=32, kernel_size=4, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(in_channels=32, out_channels=1, kernel_size=4, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(2),

        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=32, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(in_channels=32, out_channels=1, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.fcconv=nn.Linear(2992,hidden_dim*2)
        self.lstm=nn.LSTM(input_dim, hidden_dim, num_layers, dropout=dropout, bidirectional=True, batch_first=True)
        self.attention = Attention(hidden_dim, True)
        self.dropout=nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_dim * 2, output_dim)  # 输出层与输入层的维度相同

    def forward(self, inputs):
        tensorwords, seqlen = self.convert_woed2ids(inputs)
        embed_tensor = self.convert_ids2vector(tensorwords.cuda())
        data = pack_padded_sequence(embed_tensor, seqlen, batch_first=True, enforce_sorted=False).cuda()
        conv_input_data=embed_tensor#, seq_len = pad_packed_sequence(data, batch_first=True)
        conv_input_data=conv_input_data.transpose(1,2)
        conv1_out=self.conv1(conv_input_data)
        conv2_out = self.conv2(conv_input_data)
        conv3_out = self.conv3(conv_input_data)
        conv_out=torch.cat([conv1_out,conv2_out,conv3_out],-1)

        #填充确保每一个批次都一模一样
        conv_out=self.fcconv(conv_out)

        encoder_out, hidden_hc = self.lstm(data)

        encoder_hidden_output, seq_len = pad_packed_sequence(encoder_out, batch_first=True)

        atten_out = self.attention(conv_out, encoder_hidden_output,seq_len)
        feature = self.dropout(atten_out)
        output = self.linear(feature)
        return output
    def get_s(self,encoder_hidden_output,seqlen):
        batch=[]
        for i in range(len(seqlen)):
            batch.append(i)
        index=seqlen-1
        s=encoder_hidden_output[batch,index,:]
        return s.unsqueeze(1)


    def convert_woed2ids(self,words):
        listids=[]

        seqlen=[]
        for oneword in  words:
            listids_one = []
            for char in oneword:
                if char not in self.dicwk.keys():
                    listids_one.append(self.dicwk['puk'])
                else:
                    listids_one.append(self.dicwk[char])
            seqlen.append(len(listids_one))
            while(len(listids_one)<4000):
                listids_one.append(self.dicwk['pad'])
            listids.append(listids_one)
        return torch.tensor(listids),seqlen
    def convert_ids2vector(self,tensorwords):

        return self.embedder(tensorwords)
    def readidc(self,path):
        file = open(path, 'r',encoding="utf-8")
        js = file.read()
        dic = eval(js)
        file.close()
        return dic

class BiLSTM(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=128, num_layers=2, output_dim=6, dropout=0.2):
        super().__init__()
        self.dickw = self.readidc("Datasets\dick\word_kw.txt")
        self.dicwk = self.readidc("Datasets\dick\word_wk.txt")
        self.embedder = nn.Embedding(20000, input_dim)
        self.lstm=nn.LSTM(input_dim, hidden_dim, num_layers, dropout=dropout, bidirectional=True, batch_first=True)
        self.dropout=nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_dim * 2, output_dim)  # 输出层与输入层的维度相同

    def forward(self, inputs):
        tensorwords, seqlen = self.convert_woed2ids(inputs)
        embed_tensor = self.convert_ids2vector(tensorwords.cuda())
        data = pack_padded_sequence(embed_tensor, seqlen, batch_first=True, enforce_sorted=False).cuda()
        encoder_out, hidden_hc = self.lstm(data)
        encoder_hidden_output, seq_len = pad_packed_sequence(encoder_out, batch_first=True)
        feature = self.get_s(encoder_hidden_output, seq_len)
        #feature=self.dropout(feature)
        output = self.linear(feature)
        return output
    def get_s(self,encoder_hidden_output,seqlen):
        batch=[]
        for i in range(len(seqlen)):
            batch.append(i)
        index=seqlen-1
        s=encoder_hidden_output[batch,index,:]
        return s.unsqueeze(1)


    def convert_woed2ids(self,words):
        listids=[]

        seqlen=[]
        for oneword in  words:
            listids_one = []
            for char in oneword:
                if char not in self.dicwk.keys():
                    listids_one.append(self.dicwk['puk'])
                else:
                    listids_one.append(self.dicwk[char])
            seqlen.append(len(listids_one))
            while(len(listids_one)<4000):
                listids_one.append(self.dicwk['pad'])
            listids.append(listids_one)
        return torch.tensor(listids),seqlen
    def convert_ids2vector(self,tensorwords):

        return self.embedder(tensorwords)
    def readidc(self,path):
        file = open(path, 'r',encoding="utf-8")
        js = file.read()
        dic = eval(js)
        file.close()
        return dic





class LSTM(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=128, num_layers=2, output_dim=6, dropout=0.2):
        super().__init__()
        self.dickw = self.readidc("Datasets\dick\word_kw.txt")
        self.dicwk = self.readidc("Datasets\dick\word_wk.txt")
        self.embedder = nn.Embedding(20000, input_dim)
        self.lstm=nn.LSTM(input_dim, hidden_dim, num_layers, dropout=dropout, bidirectional=False, batch_first=True)
        self.dropout=nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_dim, output_dim)  # 输出层与输入层的维度相同

    def forward(self, inputs):
        tensorwords, seqlen = self.convert_woed2ids(inputs)
        embed_tensor = self.convert_ids2vector(tensorwords.cuda())
        data = pack_padded_sequence(embed_tensor, seqlen, batch_first=True, enforce_sorted=False).cuda()
        encoder_out, hidden_hc = self.lstm(data)
        encoder_hidden_output, seq_len = pad_packed_sequence(encoder_out, batch_first=True)
        feature = self.get_s(encoder_hidden_output, seq_len)
        #feature=self.dropout(feature)
        output = self.linear(feature)
        return output
    def get_s(self,encoder_hidden_output,seqlen):
        batch=[]
        for i in range(len(seqlen)):
            batch.append(i)
        index=seqlen-1
        s=encoder_hidden_output[batch,index,:]
        return s.unsqueeze(1)


    def convert_woed2ids(self,words):
        listids=[]

        seqlen=[]
        for oneword in  words:
            listids_one = []
            for char in oneword:
                if char not in self.dicwk.keys():
                    listids_one.append(self.dicwk['puk'])
                else:
                    listids_one.append(self.dicwk[char])
            seqlen.append(len(listids_one))
            while(len(listids_one)<4000):
                listids_one.append(self.dicwk['pad'])
            listids.append(listids_one)
        return torch.tensor(listids),seqlen
    def convert_ids2vector(self,tensorwords):

        return self.embedder(tensorwords)
    def readidc(self,path):
        file = open(path, 'r',encoding="utf-8")
        js = file.read()
        dic = eval(js)
        file.close()
        return dic

class Simple_RNN(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=128, num_layers=2, output_dim=6, dropout=0.2):
        super().__init__()
        self.dickw = self.readidc("Datasets\dick\word_kw.txt")
        self.dicwk = self.readidc("Datasets\dick\word_wk.txt")
        self.embedder = nn.Embedding(20000, input_dim)
        self.lstm=nn.RNN(input_dim, hidden_dim, num_layers, dropout=dropout, bidirectional=False, batch_first=True)
        self.dropout=nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_dim, output_dim)  # 输出层与输入层的维度相同

    def forward(self, inputs):
        tensorwords, seqlen = self.convert_woed2ids(inputs)
        embed_tensor = self.convert_ids2vector(tensorwords.cuda())
        data = pack_padded_sequence(embed_tensor, seqlen, batch_first=True, enforce_sorted=False).cuda()
        encoder_out, hidden_hc = self.lstm(data)
        encoder_hidden_output, seq_len = pad_packed_sequence(encoder_out, batch_first=True)
        feature = self.get_s(encoder_hidden_output, seq_len)
        #feature=self.dropout(feature)
        output = self.linear(feature)
        return output
    def get_s(self,encoder_hidden_output,seqlen):
        batch=[]
        for i in range(len(seqlen)):
            batch.append(i)
        index=seqlen-1
        s=encoder_hidden_output[batch,index,:]
        return s.unsqueeze(1)


    def convert_woed2ids(self,words):
        listids=[]

        seqlen=[]
        for oneword in  words:
            listids_one = []
            for char in oneword:
                if char not in self.dicwk.keys():
                    listids_one.append(self.dicwk['puk'])
                else:
                    listids_one.append(self.dicwk[char])
            seqlen.append(len(listids_one))
            while(len(listids_one)<4000):
                listids_one.append(self.dicwk['pad'])
            listids.append(listids_one)
        return torch.tensor(listids),seqlen
    def convert_ids2vector(self,tensorwords):

        return self.embedder(tensorwords)
    def readidc(self,path):
        file = open(path, 'r',encoding="utf-8")
        js = file.read()
        dic = eval(js)
        file.close()
        return dic