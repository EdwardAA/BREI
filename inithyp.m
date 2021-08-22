function in_hyp = inithyp(xtr,ytr,meanfunc,covfunc,likfunc,nh,hypit)
for i = 1:nh
    sn = rand(1,1);
%     disp(rng)
    num_hypers = eval(feval(covfunc{:}));
    hypers = rand(num_hypers,1);
    hyp.cov = log(hypers); hyp.mean = []; hyp.lik = log(sn);
    if i == 1
        Allh = hyp;
    end
    Allh = [Allh hyp];
    hypopt = minimize(hyp, @gp, hypit, @infGaussLik, meanfunc, covfunc, likfunc, xtr,ytr);
    try
        [y_new,~] = gpmean(xtr,xtr,ytr,hypopt,covfunc);
    catch
        [y_new,~] = gpmeannewvar(xtr,xtr,ytr,hypopt,covfunc);
    end
    m(i,1) = MSE(ytr,y_new);
end
[~,Ind] = min(m);
in_hyp = Allh(Ind+1);
  