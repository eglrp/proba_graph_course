function [mu0,mu1,sigma0,sigma1,p] = QDA(X,Y)
n0 = length(find(Y==0));
n1 = length(find(Y==1));
n = size(Y,1);
p = n1/n;
X0s = X(Y==0,:);
X1s = X(Y==1,:);
mu0 = sum(X0s)/n0;
mu1 = sum(X1s)/n1;
sigma0 = (X0s-mu0)'*(X0s-mu0) /n0;
sigma1 = (X1s-mu1)'*(X1s-mu1) /n1;
end
