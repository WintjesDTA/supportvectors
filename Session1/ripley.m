close all;
clear;
load('ripley.mat');
i1 = find(Yt ==1 );
i2 = find(Yt ==-1);
hold on;
plot(Xt(i1,1),Xt(i1,2),'ro');
plot(Xt(i2,1),Xt(i2,2),'bo');
%% 1.3 linear
hold off;
gam = 10;
[alpha,b] = trainlssvm({Xt,Yt,'c',gam,[],'lin_kernel'});
plotlssvm({Xt,Yt,'c',gam,[],'lin_kernel'},{alpha,b});
Ytest =  simlssvm({Xt,Yt,'c',gam,[],'lin_kernel'},{alpha,b},X);
sum(Y ~= Ytest)

%% RBF man tuning sig
disp('RBF kernel')
gam = 1; sig2list=[0.01, 0.1, 1, 5, 10, 25];

errlist=[];

for sig2=sig2list
    disp(['gam : ', num2str(gam), '   sig2 : ', num2str(sig2)]),
    [alpha,b] = trainlssvm({Xt,Yt,'c',gam,sig2,'RBF_kernel'});

    % Plot the decision boundary of a 2-d LS-SVM classifier
    plotlssvm({Xt,Yt,'c',gam,sig2,'RBF_kernel','preprocess'},{alpha,b});

    % Obtain the output of the trained classifier
    [Yht, Zt] = simlssvm({Xt,Yt,'c',gam,sig2,'RBF_kernel'}, {alpha,b}, X);
    err = sum(Yht~=Y); errlist=[errlist; err];
    fprintf('\n on test: #misclass = %d, error rate = %.2f%% \n', err, err/length(Y)*100)
    disp('Press any key to continue...'), pause,         
end
sig2 = .1;


%% RBF man tuning gamma
sig2 = .1;
disp('RBF kernel')
gamlist = [.2, 0.5, 1, 5, 10, 25]; 

errlist=[];

for gam =gamlist
    disp(['gam : ', num2str(gam), '   sig2 : ', num2str(sig2)]),
    [alpha,b] = trainlssvm({Xt,Yt,'c',gam,sig2,'RBF_kernel'});

    % Plot the decision boundary of a 2-d LS-SVM classifier
    plotlssvm({Xt,Yt,'c',gam,sig2,'RBF_kernel','preprocess'},{alpha,b});

    % Obtain the output of the trained classifier
    [Yht, Zt] = simlssvm({Xt,Yt,'c',gam,sig2,'RBF_kernel'}, {alpha,b}, X);
    err = sum(Yht~=Y); errlist=[errlist; err];
    fprintf('\n on test: #misclass = %d, error rate = %.2f%% \n', err, err/length(Y)*100)
    disp('Press any key to continue...'), pause,         
end

%% Tune RBF with tunelssvm
model = {Xt,Yt,'c',[1],[.1],'RBF_kernel','csa'};
[gam,sig2,cost] = tunelssvm(model,'gridsearch','crossvalidatelssvm',{10,'misclass'});
[alpha,b] = trainlssvm({Xt,Yt,'c',gam,sig2,'RBF_kernel'});
plotlssvm({Xt,Yt,'c',gam,sig2,'RBF_kernel','preprocess'},{alpha,b});
%% Tune lin with tunelssvm
model = {Xt,Yt,'c',[],[],'lin_kernel','csa'};
[gamlin,cost] = tunelssvm(model,'gridsearch','crossvalidatelssvm',{10,'misclass'});

%% Check values
close all;
%sig2 = 0.1;
%gam = 1;
[alpha,b] = trainlssvm({Xt,Yt,'c',gam,sig2,'RBF_kernel'});
[Yht, Zt] = simlssvm({Xt,Yt,'c',gam,sig2,'RBF_kernel'}, {alpha,b}, X);
plotlssvm({Xt,Yt,'c',gam,sig2,'RBF_kernel'},{alpha,b});
[alphalin,blin] = trainlssvm({Xt,Yt,'c',gamlin,[],'lin_kernel'});
%plotlssvm({Xt,Yt,'c',gamlin,[],'lin_kernel'},{alpha,b});
[Yhtlin, Ztlin] = simlssvm({Xt,Yt,'c',gamlin,[],'lin_kernel'}, {alphalin,blin}, X);
roc(Zt,Y);
title('ROC for the RBF-kernel');
roc(Ztlin,Y);
title('ROC for the lin-kernel');