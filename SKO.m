function NEI = SKO(xtr,ytr,h,covfunc,x,meanfunc,likfunc)
xnew = [xtr ; x];
try
    [y_pred, fsum] = gp(h, @infGaussLik, meanfunc, covfunc, likfunc, xtr,ytr, xnew);
catch
    [y_pred, fsum] = gpmeannewvar(xtr,xnew,ytr,h,covfunc);
end

yo = y_pred(1:end-1,:);
st = fsum(1:end-1,:);

y_pred = y_pred(end,:);
fsum = fsum(end,:);

vr = exp(2*h.lik);
u = -yo-sqrt(st);
[~,ux] = max(u);

stdv = sqrt(fsum);
fmin = ytr(ux);
diff  = fmin-y_pred;
pd = makedist('Normal');
d_stdv = (fmin-y_pred)./stdv;
skt = 1-(sqrt(vr)./sqrt(fsum));
NEI = -(((diff.*cdf(pd,d_stdv))+(stdv.*pdf(pd,d_stdv)))*skt);
