function idx = Sigmaopt_Probbandit(xtr,ytr,h,covfunc,candL,meanfunc,likfunc,z,b,R)
n = size(xtr,1);
% nit = n*10;
G = zeros(size(candL,2),1);
P = 1/size(candL,2);

[~,indx]=sort(ytr);
vin=indx(1:2,1);

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


stdv = sqrt(ycov);
fmin = min(y_train);
diff  = fmin-y_pred;
pd = makedist('Normal');
d_stdv = (fmin-y_pred)./stdv;
tr1 = (diff.^2).*cdf(pd,d_stdv);
tr2 = (2*(stdv)).*(diff.^2).*(pdf(pd,d_stdv));
tr3 = (stdv.^2).*((d_stdv.*pdf(pd,d_stdv))-1);
tr4 = ((diff.*cdf(pd,d_stdv))+(stdv.*pdf(pd,d_stdv))).^2;
mn = (diff.*cdf(pd,d_stdv))+(stdv.*pdf(pd,d_stdv));
vrnc = tr1+tr2-tr3-tr4;
for s = 1:length(candL)
    sig = candL(s);
    MI = -(mn+(sig.*sqrt(vrnc)));

    [~,ind] = min(MI);
    m2 = y_vl(ind);
    [m1,~] = min(y_train);
    gap = m1 - m2;
    G(s,1) = gap/P;
end
if z>1
    G(b,1) = 0.2*(G(b,1))+0.8*R;
end

% Pr = exp(G)./sum(exp(G));
Pr = G./sum(G);
rn = rand;
sl = zeros(size(Pr));
for ii = 1:size(Pr,1)
    if rn <= sum(Pr(1:ii))
        if ii == 1
            sl(ii) = 1;
            break;
        else
            if rn > Pr(ii-1)
                sl(ii) = 1;
                break;
            end
                
        end
    end
end
        
      
idx = find(sl);
if isempty(idx) == 1
idx = 1;
end


    
