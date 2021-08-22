function EI = ExIm2(xtr,ytr,h,covfunc,sig,x,meanfunc,likfunc)
% [y_pred,ycov] = gpmean(xtr,x,ytr,h,covfunc);
try
    [y_pred, ycov] = gp(h, @infGaussLik, meanfunc, covfunc, likfunc, xtr,ytr, x);
catch
    [y_pred,ycov] = gpmeannewvar(xtr,x,ytr,h,covfunc);
end
% fsum = diag(ycov);
stdv = sqrt(ycov);
fmin = min(ytr);
diff  = fmin-y_pred;
pd = makedist('Normal');
d_stdv = (fmin-y_pred)./stdv;
tr1 = (diff.^2).*cdf(pd,d_stdv);
tr2 = (2*(stdv)).*(diff.^2).*(pdf(pd,d_stdv));
tr3 = (stdv.^2).*((d_stdv.*pdf(pd,d_stdv))-1);
tr4 = ((diff.*cdf(pd,d_stdv))+(stdv.*pdf(pd,d_stdv))).^2;
mn = (diff.*cdf(pd,d_stdv))+(stdv.*pdf(pd,d_stdv));
vrnc = tr1+tr2-tr3-tr4;
EI = -(mn+(sig.*sqrt(vrnc)));

