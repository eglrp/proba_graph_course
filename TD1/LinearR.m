function [W,b] = LinearR(X,Y)
Xl = X;
Xl(:,end+1) = 1;
w = (Xl'*Xl)\(Xl'*Y);
W = w(1:end-1)';
b = w(end)-0.5;%because y=0 or 1, not -1 or 1
end
