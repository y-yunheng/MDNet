import numpy as np
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib


class Draw_Confusion_matrix():
    def __init__(self, matrixlen, matrixnum):
        self.matrix = []
        for i in range(matrixnum):
            one = np.zeros([matrixlen[i], matrixlen[i]])
            self.matrix.append(one)

    def compute_matrix(self, predicts, labels):
        predicts = predicts.detach().cpu()
        predicts = F.softmax(predicts, -1).numpy()
        predicts = np.argmax(predicts, -1)
        labels = labels.cpu().numpy()
        x = predicts.shape
        for k in range(predicts.shape[0]):
            for j in range(predicts.shape[1]):
                thismatrix = self.matrix[j]
                # 获取当前的预测值
                p = predicts[k][j]
                # 获取真实的预测值
                l = labels[k][j]
                if (p >= len(thismatrix)):
                    p = 0
                elif (l >= len(thismatrix)):
                    l = 0
                thismatrix[p][l] += 1

    def return_matrix(self):
        return self.matrix

    def save_matrix(self, floderpath):
        matplotlib.rcParams['font.family'] = 'STSong'  # 修改了全局变量
        matplotlib.rcParams['font.size'] = 12
        kname = 0
        for confusion_matrix in self.matrix:
            classes = [i for i in range(len(confusion_matrix))]
            plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Oranges)  # 按照像素显示出矩阵
            plt.title('')
            cb = plt.colorbar()
            # cb.set_ticks(np.linspace(0,300,20))
            tick_marks = np.arange(len(classes))
            plt.xticks(tick_marks, classes)
            plt.yticks(tick_marks, classes)
            thresh = confusion_matrix.max() / 1.
            # iters = [[i,j] for i in range(len(classes)) for j in range((classes))]
            # ij配对，遍历矩阵迭代器
            iters = np.reshape([[[i, j] for j in range(len(confusion_matrix))] for i in range(len(confusion_matrix))],
                               (confusion_matrix.size, 2))
            for i, j in iters:
                plt.text(j - 0.2, i + 0.15, format(int(confusion_matrix[i, j])))  # 显示对应的数字
            plt.xlabel('True classification')
            plt.ylabel('Forecast classification')
            # plt.text(9.5, 2.5, format('图'))
            # plt.text(9.5, 3, format('片'))
            # plt.text(9.5, 3.5, format('数'))
            # plt.text(9.5, 4, format('量'))
            # plt.text(9.5, 4.5, format('/'))
            # plt.text(9.5, 5, format('张'))
            plt.tight_layout()
            plt.savefig(floderpath + "/" + str(kname) + "confusion_matrix.svg")
            kname += 1
            plt.close()
            np.save(floderpath + "/" + str(kname) + "confusion_matrix.np", confusion_matrix)
            newnp = confusion_matrix.sum(0)
            for i in range(len(newnp)):
                if(newnp[i]!=0):
                   newnp[i] = confusion_matrix[i][i] / newnp[i]
        f = open(floderpath + "/" + "acc.txt", "w")
        a=str(newnp)
        print(str(newnp))
        f.write(a)
        print("写入cg")

