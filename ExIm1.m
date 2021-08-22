function EI = ExIm1(xtr,ytr,h,covfunc,x,meanfunc,likfunc)
% [y_pred,ycov] = gpmean(xtr,x,ytr,h,covfunc);
% [y_pred, fsum] = gp(h, @infGaussLik, meanfunc, covfunc, likfunc, xtr,ytr, x);
try
    [y_pred, fsum] = gp(h, @infGaussLik, meanfunc, covfunc, likfunc, xtr,ytr, x);
catch
    [y_pred, fsum] = gpmeannewvar(xtr,x,ytr,h,covfunc);
end
% fsum = diag(ycov);
stdv = sqrt(fsum);
fmin = min(ytr);
diff  = fmin-y_pred;
pd = makedist('Normal');
d_stdv = (fmin-y_pred)./stdv;
EI = -((diff.*cdf(pd,d_stdv))+(stdv.*pdf(pd,d_stdv)));
