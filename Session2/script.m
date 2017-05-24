clear; close all;
X = (-3:0.01:3)';
Y = sinc(X) +.1*randn(length(X),1);
figure;
plot(X,Y);
Xtrain = X(1:2:length(X));
Ytrain = Y(1:2:length(Y));
Xtest = X(2:2:length(X));
Ytest = Y(2:2:length(Y));
%% Train a first function
close all; 
gam = 100;
sig2 = 0.1;
model = {Xtrain, Ytrain, 'f', gam,sig2,'RBF_kernel'};
[alpha, b] = trainlssvm(model);
plotlssvm(model,{alpha,b});
YtestEst = simlssvm(model,{alpha,b},Xtest);
figure;
plot(Xtest,Ytest.'.');
hold on;
plot(Xtest,YtestEst,'r+');
legend('Ytest','YtestEst');

%% Tuning
gam = 100;
model = {Xtrain, Ytrain, 'f', gam,sig2,'RBF_kernel'};
cost_crossval = crossvalidate(model,10);
modeltune = {Xtrain, Ytrain, 'f', [],[],'RBF_kernel'};
[gam, sig2, cost] = tunelssvm(modeltune,'simplex','crossvalidatelssvm',{10,'mse'});

%% Train and plot
model = {Xtrain, Ytrain, 'f', gam,sig2,'RBF_kernel'};
[alpha,b] = trainlssvm(model);
plotlssvm(model,{alpha,b});

%% Bayesian framework
sig2 = .5;
gam = 10;
model = {Xtrain,Ytrain,'f',gam,sig2};
criterion_L1 = bay_lssvm(model,1);
criterion_L2 = bay_lssvm(model,2);
criterion_L3 = bay_lssvm(model,3);
[~,alpha,b] = bay_optimize(model,1);
[~,gam] = bay_optimize(model,2);
[~,sig2] = bay_optimize(model,);
sig2e = bay_errorbar({Xtrain,Ytrain,'f',gam,sig2},'figure');
%% Bayesian on Iris
load iris;
gam = .5; sig2 = 10;
bay_modoutClass({X,Y,'c',gam,sig2},'figure');
colorbar();

%% Automatic Relevance Determination
close all;
gam = 0.5;
sig2 = 0.75;
X = 6.*rand(100,3)-3;
Y = sinc(X(:,1)) + 0.1.*randn(100,1);
[selected, ranking] = bay_lssvmARD({X,Y,'class',gam,sig2});
plot(X(:,1),Y,'r+');
figure;
plot(X(:,2),Y,'bo');
figure;
plot(X(:,3),Y,'black.');






%% Robust Regression
close all;
X = (-6:0.2:6)';
Y = sinc(X) + 0.1*rand(size(X));
out = [15 17 19]';
Y(out) = 0.7+0.3*rand(size(out));
out = [41 44 46]';
Y(out) = 1.5+.2*rand(size(out));
gam = 100;
sig2 = 0.1;
model = {X,Y,'f',gam,sig2};
[alpha,b] = trainlssvm(model);
plotlssvm(model,{alpha,b});

model = initlssvm(X, Y, 'f', [],[],'RBF_kernel');
costFun = 'rcrossvalidatelssvm';
wFun = 'whuber';
model = tunelssvm(model,'simplex',costFun,{10,'mse'},wFun);
model = robustlssvm(model);
plotlssvm(model);

