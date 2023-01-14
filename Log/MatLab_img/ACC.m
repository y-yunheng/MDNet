clc
clear 

test="test_log.xlsx";
rt_test="rt_test_log.xlsx";
train="train_log.xlsx";

savedir="D:\项目\小论文\MDNet\Log\MatLab_img\img";
color=["b","g","r","k","m"];
fl=["o","x","d","v","s"];
figure(1)
for i=0:4
    modeldir="D:\项目\Graduate_project\小论文\Log\model"+i;
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
    strm=color(i+1);
    flag=fl(i+1);
    mstr=strcat("-",strm,flag);
    plot(1:N,y,mstr,"MarkerIndices",1:100:N)
    hold on
end

legend("MDNet","FastText","TextCNN","ST-MFLC", "BiLSTM")
xlabel('Step')
ylabel('ACC Value')

