function [w,b] = LDA(X,Y)
n0 = length(find(Y==0));
n1 = length(find(Y==1));
n = size(Y,1);
p = n1/n;
X0s = X(Y==0,:);
X1s = X(Y==1,:);
mu0 = sum(X0s)/n0;
mu1 = sum(X1s)/n1;
sigma = (X0s-mu0)'*(X0s-mu0) /n + (X1s-mu1)'* (X1s-mu1)/n;
w = (mu1-mu0)/sigma;
b = 0.5*(mu0/sigma*mu0' - mu1/sigma*mu1') + log(p/(1-p));
end

