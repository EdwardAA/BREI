function m = MSE(yte,pred)
b = (pred-yte).^2;
m = mean(b);