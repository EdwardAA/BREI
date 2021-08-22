function sigm = Sigmaopt_Prob(xtr,ytr,h,covfunc,candL,meanfunc,likfunc)
n = size(xtr,1);
nit = n*10;
list = zeros(size(candL,2),nit);



for i = 1:nit
    indc = randperm(n);
    xin = indc(3:end);
    vin = setdiff(indc,xin,'stable');
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
        [~,yind] = min(y_vl);
        if yind == ind
            list(s,i) = 1;
        else
            list(s,i) = 0;
        end
    end
end
sumoflist = sum(list,2);
% es = exp(sumoflist);
Pr = sumoflist./sum(sumoflist);
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
        
      
sigm = candL(find(sl));
  