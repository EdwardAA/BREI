function [X_trainset,y_trainset,yact] = data(reqitrseed,i,dim,in_tr,c,Eval_func,lb,ub,covfunc)
rng(reqitrseed(1,i),'twister')

% Train Set
xtrd = lhsdesign(in_tr,dim);
X_trainset = bsxfun(@plus,lb,bsxfun(@times,xtrd,(ub-lb)));
y_trainset = func_eval(Eval_func,X_trainset,c);
yact = func_eval(Eval_func,X_trainset,'NA');
