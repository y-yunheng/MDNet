#混淆矩阵
#confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

filepath="Log/model4/img/"
confusion_matrix = np.load(filepath+"1confusion_matrix.np.npy")

matplotlib.rcParams['font.family'] = 'STSong'  # 修改了全局变量
matplotlib.rcParams['font.size'] = 15
kname = 0
classes = [i for i in range(len(confusion_matrix))]
plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Oranges)  # 按照像素显示出矩阵
plt.title('')

cb = plt.colorbar()

tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes)
plt.yticks(tick_marks, classes)
thresh = confusion_matrix.max() / 1.
iters = np.reshape([[[i, j] for j in range(len(confusion_matrix))] for i in range(len(confusion_matrix))],
                       (confusion_matrix.size, 2))
for i, j in iters:
    maxvalue=np.max(confusion_matrix)
    if int(confusion_matrix[i, j])>0.7*maxvalue:
        plt.text(j - 0.2, i + 0.15, format(int(confusion_matrix[i, j])), color='white')  # 如果值大于最大值的百分之七十，则将文字置为白色。
    else:
        plt.text(j - 0.2, i + 0.15, format(int(confusion_matrix[i, j])))


plt.xlabel('True classification')
plt.ylabel('Predicted classification')

plt.tight_layout()
plt.savefig(filepath+"0confusion_matrix.svg")
kname += 1
plt.close()


