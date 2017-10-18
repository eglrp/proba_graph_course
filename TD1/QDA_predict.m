function [Y] = QDA_predict(X,mu0,mu1,sigma0,sigma1,p)
Xc0 = X - mu0;
Xc1 = X - mu1;
a = Xc0/sigma0;
a = a.*Xc0;
a = sum(a,2);
b = Xc1/sigma1;
b = b.*Xc1;
b = sum(b,2);
u = -0.5*(a-b);
Y = (u<log(p/(1-p)))* 1.0;
end

