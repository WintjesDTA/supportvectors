%% 1.1
x1 = 1 + randn(50,2);
x2 = -1 + randn(51,2);
y1 = ones(50,1);
y2 = -ones(51,1);

X = [x1;x2];
Y = [y1;y2];

hold on;
plot(x1(:,1),x1(:,2),'ro');
plot(x2(:,1),x2(:,2),'bo');
% Draw line with line([-1.5, 1.5],[3,-2])

%% 1.3 linear
hold off;
gam = 10;
load('iris')
[alpha,b] = trainlssvm({X,Y,'c',gam,[],'lin_kernel'});
plotlssvm({X,Y,'c',gam,[],'lin_kernel'},{alpha,b});
Ytest =  simlssvm({X,Y,'c',gam,[],'lin_kernel'},{alpha,b},Xt);
result = Yt == Ytest;
sum(result)

%% 1.3 polynomial
gam = 5;
t = 1;
degree = 4;
Ytest = tptpoly(X,Y,Xt,gam,t,degree);
result = Ytest == Yt;
sum(result)

%% 1.4 RBF
%  See SampleScript

%% Hyperparameters
close all
gamlist = [1,10,100];
sig2list = [.1,.5,1,2,3,5,7,10];
errlist = [];
len = size(sig2list);
for gam = gamlist
    for sig2 = sig2list
        idx = randperm(size(X,1));
        Xtrain = X(idx(1:80),:);
        Ytrain = Y(idx(1:80),:);
        Xval = X(idx(81:100),:);
        Yval = Y(idx(81:100),:);
        [alpha,b] = trainlssvm({Xtrain,Ytrain,'c',gam,sig2, ...
        'RBF_kernel'});
        estYval = simlssvm({Xtrain,Ytrain,'c',gam,sig2,'RBF_kernel'}, ...
        {alpha,b},Xval);
        error = 20 -sum(estYval == Yval);
        errlist = [errlist, error];
    end
end
figure;
hold on;
title('Error plot')
xlabel('Sigma')
ylabel('Misclassifications')
xlab = sig2list;
plot(xlab,errlist(1:len(2)),'b');
plot(xlab,errlist(len(2)+1:2*len(2)),'g');
plot(xlab,errlist(2*len(2)+1:end),'red');
legend('gamma = 1','gamma = 10','gamma = 100');
hold off;

%% Crossvalidation
close all
gamlist = [1,10,100];
sig2list = [.1,.5,1:.5:10];
perflist = [];
len = size(sig2list);
for gam = gamlist
    for sig2 = sig2list
        performance = crossvalidate({X,Y,'c',gam,sig2,'RBF_kernel'}, 10, 'misclass');
        perflist = [perflist, performance];
    end
end
figure;
hold on;
title('Cost plot')
xlabel('Sigma')
ylabel('Cost')
xlab = sig2list;
plot(xlab,perflist(1:len(2)),'b');
plot(xlab,perflist(2*len(2)+1:end),'red');
plot(xlab,perflist(len(2)+1:2*len(2)),'g');
legend('gamma = 1','gamma = 10','gamma = 100');
hold off;

%% Tune parameters
fprintf('startin here');
model = {X,Y,'c',[],[],'RBF_kernel','csa'};
[gam,sig2,cost] = tunelssvm(model,'gridsearch','crossvalidatelssvm',{10,'misclass'});

%ROC curve
fprintf(num2str(gam));
[alpha,beta] = trainlssvm({Xtrain,Ytrain,'c',gam,sig2,'RBF_kernel'});
[Ysim, Ylatent] = simlssvm({Xtrain,Ytrain,'c',gam,sig2, 'RBF_kernel'},{alpha,beta},Xval);
roc(Ylatent,Yval);
