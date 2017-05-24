clear; close all;
load logmap
order = 10;
X = windowize(Z,1:(order+1));
Y = X(:,end);
X = X(:,1:order);
gam = 10; sig2 = 10;
model = {X,Y,'f',gam,sig2};
[alpha,b,cost] = trainlssvm(model);
plotlssvm(model,{alpha,b});

%%
order = 5;
X = windowize(Z,1:(order+1));
Y = X(:,end);
X = X(:,1:order);
model = {X,Y,'f',[],[],'RBF_kernel','csa'};
[gam,sig2,cost] = tunelssvm(model,'simplex','crossvalidatelssvm',{10,'mse'});
[alpha,b] = trainlssvm({X,Y,'f',gam,sig2});
plotlssvm({X,Y,'f',gam,sig2},{alpha,b});
performance = crossvalidate({X,Y,'f',gam,sig2,'RBF_kernel'});

%% Tune parameters

orderList = 5:25;
costList = zeros(length(orderList),1);
for i = 1:length(orderList)
    X = windowize(Z,1:(orderList(i)+1));
    Y = X(:,end);
    X = X(:,1:orderList(i));
    model = {X,Y,'f',[],[],'RBF_kernel','csa'};
    [gam,sig2] = tunelssvm(model,'simplex','crossvalidatelssvm',{10,'mse'});
    costnumberi = crossvalidate({X,Y,'f',gam,sig2,'RBF_kernel'});    
    costList(i,1) = costnumberi;    
end
plot(orderList,costList);
xlabel('order'); ylabel('cost');
%% Predict
order = 10;
X = windowize(Z,1:(order+1));
Y = X(:,end);
X = X(:,1:order);
model = {X,Y,'f',[],[],'RBF_kernel','csa'};
[gam,sig2] = tunelssvm(model,'simplex','crossvalidatelssvm',{10,'mse'});
[alpha,b] = trainlssvm({X,Y,'f',gam,sig2});



%% Predict multiple
order =10;
X = windowize(Z,1:(order+1));
Y = X(:,end);
X = X(:,1:order);
horizon = length(Ztest)-order;
Zpt = predict({X,Y,'f',gam,sig2},Ztest(1:order),horizon);
plot([Ztest(order+1:end) Zpt]);
error = Ztest(order+1:end)-Zpt;
mse = sum(error.^2)/(length(error))