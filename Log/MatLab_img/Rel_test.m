clc
clear 

test="test_log.xlsx";
rt_test="rt_test_log.xlsx";
train="train_log.xlsx";

savedir="D:\项目\小论文\MDNet\Log\MatLab_img\img";
color=["r","g","b","b","m","k","c"];
fl=["o","x","d","v","s","+","p","h"];
as1=[0, 1, 2,3, 4,5,6];
figure(1)
for i=1:7
    modeldir="D:\项目\小论文\MDNet\Log\model"+as1(i);
    model_rt_test=readcell(modeldir+"/"+rt_test);    
    %if(i==1)
    %    continue
    %end
    N=size(model_rt_test,1)-1;
    ystr=model_rt_test(:,3);
    % 将每个字符串转换为数值
    y = zeros(size(ystr)-1);
    for k = 2:numel(ystr)
        y(k-1) = str2double(ystr{k}(2:end-1));
    end
    
    strm=color(i);
    flag=fl(i);
    mstr=strcat("-",strm,flag);
    plot(1:N,y,mstr,"MarkerIndices",1:1:N)
    hold on
end
legend("C-PsyD","FastText","TextCNN","ST-MFLC", "BiLSTM","LSTM","Simple-RNN")
xlabel('Epoch')
ylabel('Eva ACC')