function c = add_noise(dim,lb,ub,noise_level,Eval_func)
xtr = lhsdesign(1000,dim);
X_train = bsxfun(@plus,lb,bsxfun(@times,xtr,(ub-lb)));
ytrain = func_eval(Eval_func,X_train,'NA');
m = mean(ytrain);
c = (noise_level*m)/100;