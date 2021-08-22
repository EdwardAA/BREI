function [Mn,cv] = gpmeannewvar2(xtr,xte,yt,h,covfunc)
vr = exp(2*h.lik);
kxx = feval(covfunc{:}, h.cov, xtr, xtr);
knx = feval(covfunc{:}, h.cov, xte, xtr);
knn = feval(covfunc{:}, h.cov, xte, xte);
kxn = knx';
I = eye(size(kxx));
I = (vr)*I;
V = kxx+I;
n = length(kxx);
m = length(knn);
% sn2 = exp(2*h.lik);
% W = ones(n,1)/sn2; 
% sW = sqrt(W);
try
    L = chol(V,'lower');
catch
    try
        addvar = 1e-10*abs(max(max(kxx)));
        Vnew = V+eye(n)*addvar;
        L = chol(Vnew,'lower');
    catch
        addvar = 1e-5*abs(max(max(kxx)));
        Vnew = V+eye(n)*addvar;
        L = chol(Vnew,'lower');
    end
        
end

Z = (L\kxn);
% H = inv(V);
if isnan(yt) == 1
    Mn = nan;
else
    alpha = L'\(L\yt);
    Mn = (knx)*(alpha);
end
cv = knn - (Z'*Z);

cvm = diag(cv);
cvm = max(cvm,0);
cvm = cvm +vr;

cv(1:m+1:end) = cvm;

% knn = diag(knn);
% 
% 
% 
% 
% V  = L'\(repmat(sW,1,length(knn)).*kxn);
% cv = knn - sum(V.*V,1)';  
% cv = max(cv,0);
% cv = cv + sn2; 

