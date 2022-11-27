import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropyCriterion(nn.Module):
    def __init__(self):
        super(CrossEntropyCriterion, self).__init__()
        self.myloss=torch.nn.CrossEntropyLoss()
    def forward(self, predict, label):
        #input(N,C,d1,d2,dk.....)  label(N,d1,d2,dk....)
        predict=torch.transpose(predict,-2,-1)
        res=self.myloss(predict,label)
        return res
