clc
clear 

test="test_log.xlsx";
rt_test="rt_test_log.xlsx";
train="train_log.xlsx";

savedir="D:\项目\小论文\MDNet\Log\img";
color=["r","g","b","c","m","b","k","c"];
fl=["o","x","d","v","s","+","p","h"];
as1=[0, 1, 2,3,4,5,6];
figure(1)
for i=1:7
    modeldir="D:\项目\小论文\MDNet\Log\model"+as1(i);
    %model_test=readmatrix(modeldir+"/"+test);
    model_train=readmatrix(modeldir+"\"+train);
    %model_rt_test=readmatrix(modeldir+"/"+rt_test);    
    %if(i==1)
    %    continue
    %end
    N=size(model_train,1);
    y=model_train(:,5);
    for j=1:50
        y=smooth(y);
    end
    strm=color(i);
    flag=fl(i);
    mstr=strcat("-",strm,flag);
    plot(1:N,y,mstr,"MarkerIndices",1:50:N)
    hold on
end

legend("C-PsyD","FastText","TextCNN","ST-MFLC", "BiLSTM","LSTM","Simple-RNN")
xlabel('Step')
ylabel('ACC')

