function [Ytest] = tptpoly(X,Y,Xt,gam,t,degree)
%TPTPOLY Summary of this function goes here
%   Detailed explanation goes here
    [alpha,b] = trainlssvm({X,Y,'c',gam,[t;degree],'poly_kernel'});    
    plotlssvm({X,Y,'c',gam,[t;degree],'poly_kernel'},{alpha,b});
    Ytest = simlssvm({X,Y,'c',gam,[1;degree],'poly_kernel'},{alpha,b},Xt);   
end

