import numpy as np
import torch
import torch.nn.functional as F

class myPerplexity():
    def __init__(self, name='MutiAcc', *args, **kwargs):
        super(myPerplexity, self).__init__(*args, **kwargs)
        self._name = name
        self.total_ce = 0
        self.total_word_num = 0

    def compute(self, pred, label, seq_len,seq_mask):
        """
        Computes cross entropy loss.

        Args:
            pred (Tensor):
                Predictor tensor, and its dtype is float32 or float64, and has
                a shape of [batch_size, sequence_length, vocab_size].
            label(Tensor):
                Label tensor, and its dtype is int64, and has a shape of
                [batch_size, sequence_length, 1] or [batch_size, sequence_length].
            seq_mask(Tensor, optional):
                Sequence mask tensor, and its type could be float32, float64,
               int32 or int64, and has a shape of [batch_size, sequence_length].
                It's used to calculate loss. Defaults to None.

        """
        if label.dim() == 2:
            label = torch.unsqueeze(label, axis=2)
        ce = F.cross_entropy(
            input=pred, label=label, reduction='none', soft_label=False)
        ce = torch.squeeze(ce, axis=[2])
        if seq_mask is not None:
            ce = ce * seq_mask
            word_num = torch.sum(seq_mask)
            return ce, word_num
        return ce

    def update(self, ce, word_num=None):
        """
        Updates metric states.

        Args:
            ce (numpy.ndarray):
                Cross entropy loss, it's calculated by `compute` and converted
                to `numpy.ndarray`.
            word_num (numpy.ndarray):
               The number of words of sequence, it's calculated by `compute`
               and converted to `numpy.ndarray`. Defaults to None.

        """
        batch_ce = torch.sum(torch.to_tensor(ce))
        if word_num is None:
            word_num = ce.shape[0] * ce.shape[1]
        else:
            word_num = word_num[0]
        self.total_ce += batch_ce
        self.total_word_num += word_num

    def reset(self):
        """
        Resets all metric states.
        """
        self.total_ce = 0
        self.total_word_num = 0

    def accumulate(self):
        """
        Calculates and returns the value of perplexity.

        Returns:
            perplexity: Calculation results.
        """
        return torch.exp(self.total_ce / self.total_word_num).numpy()[0]

    def name(self):
        """
        Returns name of the metric instance.

        Returns:
           str: The name of the metric instance.

        """
        return self._name

    def getacc(self,predicts, label, label_len, maskl):
        self.reset()
        ce, xc = self.compute(predicts, label, label_len, maskl)
        self.update(ce, xc)
        acc = self.accumulate()
        return acc


class MutiAccuracy():

    def __init__(self, name="acc", *args, **kwargs):
        super(MutiAccuracy, self).__init__(*args, **kwargs)
        self.alle=np.array([0],dtype='float32')
        self.allc=np.array([0],dtype='float32')
        self._init_name(name)
        self.reset()

    def compute(self, pred, label):
        arg_pred=torch.argmax(pred,-1)
        nowall=np.array([0],dtype='float32')
        nowacc=np.array([0],dtype='float32')
        for  i in range(arg_pred.shape[0]):
            for j in range(arg_pred.shape[1]):
                if arg_pred[i][j]==label[i][j]:
                    nowacc[j]+=1
                nowall[j]+=1

        ce=np.array([nowacc,nowall])
        return ce

    def update(self, corret):
        """
        Update the metrics states (correct count and total count), in order to
        calculate cumulative accuracy of all instances. This function also
        returns the accuracy of current step.

        Args:
            correct: Correct mask, a tensor with shape [batch_size, d0, ..., topk].
        Return:
            Tensor: the accuracy of current step.
        """
        self.allc+=corret[0]
        self.alle+=corret[1]


    def reset(self):
        """
        Resets all of the metric state.
        """
        self.alle =np.array([0],dtype='float32')
        self.allc =np.array([0],dtype='float32')

    def accumulate(self):
        """
        Computes and returns the accumulated metric.
        """

        return self.allc/self.alle
    def getacc(self,predicts, label):
        ce= self.compute(predicts, label)
        self.update(ce)
        acc = self.accumulate()
        return acc
    def _init_name(self, name):
        self._name = [name]

    def name(self):
        """
        Return name of metric instance.
        """
        return self._name