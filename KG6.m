function NEI = KG6(xtr,ytr,h,covfunc,x,zval)

xnew = [xtr ; x];
[yc, fc] = gpmeannewvar2(xtr,xnew,ytr,h,covfunc);

vr = exp(2*h.lik);

Av = zeros(length(zval),1);
% knew = feval(covfunc{:}, h.cov, xtr, x);
fsum = fc(:,end);
ssq = fc(end,end);
m = min(yc);
ratio =fsum./sqrt(ssq+vr);
for j = 1:length(zval)
    t2 = yc + (ratio.*zval(j,1));
    Av(j,1) = m-(min(t2));
%     Av(j,1) = min(t2);
end

NEI = -(mean(Av));
% NEI = -(m - mean(Av));
