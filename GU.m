function G = GU(xtr,ytr,h,covfunc,x,meanfunc,likfunc,delta,t)

try
    [y_pred, fsum] = gp(h, @infGaussLik, meanfunc, covfunc, likfunc, xtr,ytr, x);
catch
    [y_pred, fsum] = gpmeannewvar(xtr,x,ytr,h,covfunc);
end
s = sqrt(fsum);
d = size(xtr,2);
b = 2*(log((t^(d/2+2))*(pi^2)/(3*delta)));
%b = 2*(log((t^2)*(pi^2)/(6*delta)));
b = b/5;
b = sqrt(b);
G = (y_pred+(b*s));
