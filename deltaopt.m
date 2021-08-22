function gap = deltaopt(xtr,ytr,h,covfunc,meanfunc,likfunc,delta,t)
n = size(xtr,1);
[~,indx]=sort(ytr);
q = 0.2*(n);
vin=indx(1:q,1);

indc = randperm(n);
xin = setdiff(indc,vin,'stable');
X_train = xtr(xin,:);
y_train = ytr(xin,:);
X_vl = xtr(vin,:);
y_vl = ytr(vin,:);
try
    [y_pred, ycov] = gp(h, @infGaussLik, meanfunc, covfunc, likfunc, X_train,y_train, X_vl);
catch
    [y_pred, ycov] = gpmeannewvar(X_train,X_vl,y_train,h,covfunc);
    disp('Positive Definite Problem')
end


s = sqrt(ycov);
d = size(xtr,2);
b = 2*(log((t^(d/2+2))*(pi^2)/(3*delta)));
%b = 2*(log((t^2)*(pi^2)/(6*delta)));
b = sqrt(b);
G = (y_pred+(b*s));
[~,ind] = min(G);
m2 = y_vl(ind);
[m1,~] = min(y_train);
gap = -(m1 - m2);



    
