#混淆矩阵
#confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd

modelsname=[ "C-PsyD", "FastText", "TextCNN","ST-MFLC", "BiLSTM","LSTM","Simple-RNN"]
writer = pd.ExcelWriter("混淆矩阵表格.xlsx", engine='openpyxl')
syncdir=r"D:\BaiduNetdiskWorkspace\OneDrive\论文\硕士毕业论文\小论文-心理学分类\img"
for indexi in range(len(modelsname)):
    filepath = "Log/model" + str(indexi) + "/img/"
    confusion_matrix = np.load(filepath + "1confusion_matrix.np.npy")
    # 将npy数组转换为pandas DataFrame
    df = pd.DataFrame(confusion_matrix)

    df.to_excel(writer, index=False, sheet_name=modelsname[indexi])

    confusion_matrix = np.delete(confusion_matrix, obj=0, axis=0)
    confusion_matrix = np.delete(confusion_matrix, obj=0, axis=1)
    matplotlib.rcParams['font.family'] = 'STSong'  # 修改了全局变量
    matplotlib.rcParams['font.size'] = 15
    kname = 0
    classes = [i + 1 for i in range(len(confusion_matrix))]
    plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)  # 按照像素显示出矩阵
    plt.title('')

    cb = plt.colorbar()

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    thresh = confusion_matrix.max() / 1.
    iters = np.reshape([[[i, j] for j in range(len(confusion_matrix))] for i in range(len(confusion_matrix))],
                       (confusion_matrix.size, 2))
    for i, j in iters:
        maxvalue = np.max(confusion_matrix)
        if int(confusion_matrix[i, j]) > 0.7 * maxvalue:
            plt.text(j - 0.2, i + 0.15, format(int(confusion_matrix[i, j])), color='white')  # 如果值大于最大值的百分之七十，则将文字置为白色。
        else:
            plt.text(j - 0.2, i + 0.15, format(int(confusion_matrix[i, j])))

    plt.title(modelsname[indexi])
    plt.xlabel('Correct  Classification')
    plt.ylabel('Forecast Classification')

    plt.tight_layout()
    plt.savefig(filepath + modelsname[indexi] + "混淆矩阵.svg")
    plt.savefig(syncdir + "/Figure "+str(indexi+14) +"Confusion matrix obtained by "+modelsname[indexi]+ ".svg")
    # plt.savefig("D:/BaiduNetdiskWorkspace/OneDrive/论文/硕士毕业论文/毕业论文/图像/"+ modelsname[indexi]+"混淆矩阵.svg")
    kname += 1
    plt.close()



writer.save()
writer.close()