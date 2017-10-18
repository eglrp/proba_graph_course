function [Y] = predict(X,w,b)
Y = X * w' + b > 0.0;
Y = Y * 1;
end

