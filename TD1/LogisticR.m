function [W,b] = LogisticR(X,Y,init_w,init_b)
Xl = X;
Xl(:,end+1) = 1;
w = vertcat(init_w',init_b);
for i=1:100
    eta = 1./(1+exp(-Xl*w));
    D = eta.*(1-eta);
    w = w + (Xl'*diag(D)*Xl) \ (Xl'*(Y-eta));
end
W = w(1:end-1)';
b = w(end);
end

