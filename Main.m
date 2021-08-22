syed
clc; clear;
% addpath(genpath('/home-new/jmx397/gpml-matlab-v4.2-2018-06-11'))
rng(1,'twister')
%% User Inputs - Changed Every run
D = 2;
Eval_func = 'Levy';
in_tr = 10*D;  % initial training set
noise_level = 0;
candL = [-0.75,-0.5,-0.25,0,0.25,0.5,0.75];
zn = 100; % For KG 
%% Other user Inputs that are constant
nit = 1000;
req = 100; %Num repetetions
hypit = -1000; %GPML input - Max Eval for hyp opt
nh = 10;
q = 100;   %Num of additional evaluations                                                 
[lb,ub] = func_bounds(Eval_func,D);
c = add_noise(D,lb,ub,noise_level,Eval_func);

%% GPML Mean, Cov ang Lik functions
meanfunc = [];                  
covfunc = {'covSEiso'}; 
likfunc = @likGauss;
%%
itrSeed = randi(1000,1,nit);
reqitrseed = itrSeed(1:req);
%% Main

%% EI
methodname = 'EI';
yobjval = zeros(q+1,req);
times = zeros(q,req);
sigvl = zeros(q,req);
filename = strcat(Eval_func,'_',num2str(in_tr),'_',num2str(req),'_',num2str(noise_level),'_',methodname,'.csv');

parfor i = 1:req

    res_time = zeros(q,1);
    yobj = zeros(q+1,1);

    [X_trainset,y_trainset,yact] = data(reqitrseed,i,D,in_tr,c,Eval_func,lb,ub);

    yobj(1) = min(yact);
    in_hyp = inithyp(X_trainset,y_trainset,meanfunc,covfunc,likfunc,nh,hypit);

    % Hyper parameter optimization function
    hyp2 = minimize(in_hyp, @gp, hypit, @infGaussLik, meanfunc, covfunc, likfunc, X_trainset,y_trainset);

    X_train = X_trainset;
    y_train = y_trainset;
    hyp1 = hyp2;
   
    for z = 1:q
 
        tic;
        fun2 = @(x) ExIm1(X_train,y_train,hyp1,covfunc,x,meanfunc,likfunc);
        [X_add,einew,~,~] = particleswarm(fun2,D,lb,ub);
        lt = toc;        
        y_add = func_eval(Eval_func,X_add,c);
        ytrue = func_eval(Eval_func,X_add,'NA');

        X_train = vertcat(X_train, X_add);
        y_train = vertcat(y_train, y_add);
        yact = vertcat(yact, ytrue);
        hyp1 = minimize(hyp1, @gp, hypit, @infGaussLik, meanfunc, covfunc, likfunc, X_train,y_train);
        ybest = min(yact);
        yobj(z+1) = ybest;
        res_time(z) = lt;


    end

    times(:,i) = res_time;
    yobjval(:,i) = yobj;
    
end

% Get Final Average MSE 
Final_time = mean(times,2);
Final_yobj = mean(yobjval,2);
alld = zeros((2*(q+1))+1,(req*3));
alld(q+4:(2*(q+1))+1,1:req) = times;
alld(2:q+1,1) = Final_time;
alld(q+4:(2*(q+1))+1,req+1:(2*req)) = sigvl;
alld(q+3:(2*(q+1))+1,(2*req)+1:(3*req)) = yobjval;
alld(1:q+1,2) = Final_yobj;

row_header1 = zeros(q+1,1);
row_header1(1:2*(q+1)+1,1) = (linspace(0,(2*q)+2,(2*q)+3))';

alld = [row_header1,alld];
for k = 1:3
    if k == 1
        headernew{k} = ['Num'];
    elseif k == 2
        headernew{k} = ['Avg_time'];
    elseif k == 3
        headernew{k} = ['Avg_Yobj'];
    end
end
     

cHeader = headernew;
commaHeader = [cHeader;repmat({','},1,numel(cHeader))]; %insert commaas
commaHeader = commaHeader(:)';
textHeader = cell2mat(commaHeader); %cHeader in text with commas
%write header to file
fid = fopen(filename,'w'); 
fprintf(fid,'%s\n',textHeader);
fclose(fid);
%write data to end of file
dlmwrite(filename,alld,'-append');
Edata = Final_yobj;
Etime = Final_time;
%% BREI
methodname = 'BREI';
yobjval = zeros(q+1,req);
times = zeros(q,req);
sigvl = zeros(q+1,req);
filename = strcat(Eval_func,'_',num2str(in_tr),'_',num2str(req),'_',num2str(noise_level),'_',methodname,'.csv');


parfor i = 1:req

    res_time = zeros(q,1);
    yobj = zeros(q+1,1);
    sg = zeros(q+1,1);

    [X_trainset,y_trainset,yact] = data(reqitrseed,i,D,in_tr,c,Eval_func,lb,ub);

    yobj(1) = min(yact);
    in_hyp = inithyp(X_trainset,y_trainset,meanfunc,covfunc,likfunc,nh,hypit);

    % Hyper parameter optimization function
    hyp2 = minimize(in_hyp, @gp, hypit, @infGaussLik, meanfunc, covfunc, likfunc, X_trainset,y_trainset);


    X_train = X_trainset;
    y_train = y_trainset;
    hyp1 = hyp2;
   
    for z = 1:q
        tic;
        if z == 1
            R = 0;
            b = 0;
            indx1 = Sigmaopt_Probbandit(X_train,y_train,hyp1,covfunc,candL,meanfunc,likfunc,z,b,R);
        else
            R = yobj(z-1)-yobj(z);
            indx1 = Sigmaopt_Probbandit(X_train,y_train,hyp1,covfunc,candL,meanfunc,likfunc,z,indx1,R);
        end

        sig =  candL(indx1);
        sg(z+1,1) = sig;
        fun2 = @(x) ExIm2(X_train,y_train,hyp1,covfunc,sig,x,meanfunc,likfunc);
       
        [X_add,einew,~,~] = particleswarm(fun2,D,lb,ub);
        lt = toc;        
        y_add = func_eval(Eval_func,X_add,c);
        ytrue = func_eval(Eval_func,X_add,'NA');

        X_train = vertcat(X_train, X_add);
        y_train = vertcat(y_train, y_add);
        yact = vertcat(yact, ytrue);
        hyp1 = minimize(hyp1, @gp, hypit, @infGaussLik, meanfunc, covfunc, likfunc, X_train,y_train);
        ybest = min(yact);
        yobj(z+1) = ybest;
        res_time(z) = lt;


    end

    times(:,i) = res_time;
    yobjval(:,i) = yobj;
    sigvl(:,i) = sg;
    
end

% Get Final Average MSE 
Final_time = mean(times,2);
Final_yobj = mean(yobjval,2);
alld = zeros((2*(q+1))+1,(req*3));
alld(q+4:(2*(q+1))+1,1:req) = times;
alld(2:q+1,1) = Final_time;
alld(q+3:(2*(q+1))+1,req+1:(2*req)) = sigvl;
alld(q+3:(2*(q+1))+1,(2*req)+1:(3*req)) = yobjval;
alld(1:q+1,2) = Final_yobj;

row_header1 = zeros(q+1,1);
row_header1(1:2*(q+1)+1,1) = (linspace(0,(2*q)+2,(2*q)+3))';

alld = [row_header1,alld];
for k = 1:3
    if k == 1
        headernew{k} = ['Num'];
    elseif k == 2
        headernew{k} = ['Avg_time'];
    elseif k == 3
        headernew{k} = ['Avg_Yobj'];
    end
end
     

cHeader = headernew;
commaHeader = [cHeader;repmat({','},1,numel(cHeader))]; %insert commaas
commaHeader = commaHeader(:)';
textHeader = cell2mat(commaHeader); %cHeader in text with commas
%write header to file
fid = fopen(filename,'w'); 
fprintf(fid,'%s\n',textHeader);
fclose(fid);
%write data to end of file
dlmwrite(filename,alld,'-append');
Bdata = Final_yobj;
Btime = Final_time;
%% SKO
methodname = 'SKO';
yobjval = zeros(q+1,req);
times = zeros(q,req);
sigvl = zeros(q,req);
filename = strcat(Eval_func,'_',num2str(in_tr),'_',num2str(req),'_',num2str(noise_level),'_',methodname,'.csv');

parfor i = 1:req
    res_time = zeros(q,1);
    yobj = zeros(q+1,1);

    [X_trainset,y_trainset,yact] = data(reqitrseed,i,D,in_tr,c,Eval_func,lb,ub);

    yobj(1) = min(yact);
    in_hyp = inithyp(X_trainset,y_trainset,meanfunc,covfunc,likfunc,nh,hypit);

    % Hyper parameter optimization function
    hyp2 = minimize(in_hyp, @gp, hypit, @infGaussLik, meanfunc, covfunc, likfunc, X_trainset,y_trainset);


    X_train = X_trainset;
    y_train = y_trainset;
    hyp1 = hyp2;
   
    for z = 1:q
 
        tic;
        fun2 = @(x) SKO(X_train,y_train,hyp1,covfunc,x,meanfunc,likfunc);
        [X_add,einew,~,~] = particleswarm(fun2,D,lb,ub);
        lt = toc;        
        y_add = func_eval(Eval_func,X_add,c);
        ytrue = func_eval(Eval_func,X_add,'NA');

        X_train = vertcat(X_train, X_add);
        y_train = vertcat(y_train, y_add);
        yact = vertcat(yact, ytrue);
        hyp1 = minimize(hyp1, @gp, hypit, @infGaussLik, meanfunc, covfunc, likfunc, X_train,y_train);
        ybest = min(yact);
        yobj(z+1) = ybest;
        res_time(z) = lt;


    end

    times(:,i) = res_time;
    yobjval(:,i) = yobj;
    
end

% Get Final Average MSE 
Final_time = mean(times,2);
Final_yobj = mean(yobjval,2);
alld = zeros((2*(q+1))+1,(req*3));
alld(q+4:(2*(q+1))+1,1:req) = times;
alld(2:q+1,1) = Final_time;
alld(q+4:(2*(q+1))+1,req+1:(2*req)) = sigvl;
alld(q+3:(2*(q+1))+1,(2*req)+1:(3*req)) = yobjval;
alld(1:q+1,2) = Final_yobj;

row_header1 = zeros(q+1,1);
row_header1(1:2*(q+1)+1,1) = (linspace(0,(2*q)+2,(2*q)+3))';

alld = [row_header1,alld];
for k = 1:3
    if k == 1
        headernew{k} = ['Num'];
    elseif k == 2
        headernew{k} = ['Avg_time'];
    elseif k == 3
        headernew{k} = ['Avg_Yobj'];
    end
end
     

cHeader = headernew;
commaHeader = [cHeader;repmat({','},1,numel(cHeader))]; %insert commaas
commaHeader = commaHeader(:)';
textHeader = cell2mat(commaHeader); %cHeader in text with commas
%write header to file
fid = fopen(filename,'w'); 
fprintf(fid,'%s\n',textHeader);
fclose(fid);
%write data to end of file
dlmwrite(filename,alld,'-append');
Sdata = Final_yobj;
Stime = Final_time;
%% KG
methodname = 'KG';
yobjval = zeros(q+1,req);
times = zeros(q,req);
sigvl = zeros(q,req);
filename = strcat(Eval_func,'_',num2str(in_tr),'_',num2str(req),'_',num2str(noise_level),'_',methodname,'.csv');


parfor i = 1:req

    res_time = zeros(q,1);
    yobj = zeros(q+1,1);

    [X_trainset,y_trainset,yact] = data(reqitrseed,i,D,in_tr,c,Eval_func,lb,ub);

    yobj(1) = min(yact);
    in_hyp = inithyp(X_trainset,y_trainset,meanfunc,covfunc,likfunc,nh,hypit);

    % Hyper parameter optimization function
    hyp2 = minimize(in_hyp, @gp, hypit, @infGaussLik, meanfunc, covfunc, likfunc, X_trainset,y_trainset);


    X_train = X_trainset;
    y_train = y_trainset;
    hyp1 = hyp2;
   
    for z = 1:q
        zval = randn(zn,1); 
        tic;
        fun2 = @(x) KG6(X_train,y_train,hyp1,covfunc,x,zval);
        [X_add,einew,~,~] = particleswarm(fun2,D,lb,ub);
        lt = toc;        
        y_add = func_eval(Eval_func,X_add,c);
        ytrue = func_eval(Eval_func,X_add,'NA');

        X_train = vertcat(X_train, X_add);
        y_train = vertcat(y_train, y_add);
        yact = vertcat(yact, ytrue);
        hyp1 = minimize(hyp1, @gp, hypit, @infGaussLik, meanfunc, covfunc, likfunc, X_train,y_train);
        ybest = min(yact);
        yobj(z+1) = ybest;
        res_time(z) = lt;


    end

    times(:,i) = res_time;
    yobjval(:,i) = yobj;
    
end

% Get Final Average MSE 
Final_time = mean(times,2);
Final_yobj = mean(yobjval,2);
alld = zeros((2*(q+1))+1,(req*3));
alld(q+4:(2*(q+1))+1,1:req) = times;
alld(2:q+1,1) = Final_time;
alld(q+4:(2*(q+1))+1,req+1:(2*req)) = sigvl;
alld(q+3:(2*(q+1))+1,(2*req)+1:(3*req)) = yobjval;
alld(1:q+1,2) = Final_yobj;

row_header1 = zeros(q+1,1);
row_header1(1:2*(q+1)+1,1) = (linspace(0,(2*q)+2,(2*q)+3))';

alld = [row_header1,alld];
for k = 1:3
    if k == 1
        headernew{k} = ['Num'];
    elseif k == 2
        headernew{k} = ['Avg_time'];
    elseif k == 3
        headernew{k} = ['Avg_Yobj'];
    end
end
     

cHeader = headernew;
commaHeader = [cHeader;repmat({','},1,numel(cHeader))]; %insert commaas
commaHeader = commaHeader(:)';
textHeader = cell2mat(commaHeader); %cHeader in text with commas
%write header to file
fid = fopen(filename,'w'); 
fprintf(fid,'%s\n',textHeader);
fclose(fid);
%write data to end of file
dlmwrite(filename,alld,'-append');
Kdata = Final_yobj;
Ktime = Final_time;
%% GPUCB
methodname = 'GPUCB';
yobjval = zeros(q+1,req);
times = zeros(q,req);
sigvl = zeros(q,req);
filename = strcat(Eval_func,'_',num2str(in_tr),'_',num2str(req),'_',num2str(noise_level),'_',methodname,'.csv');

parfor i = 1:req

    res_time = zeros(q,1);
    yobj = zeros(q+1,1);

    [X_trainset,y_trainset,yact] = data(reqitrseed,i,D,in_tr,c,Eval_func,lb,ub);

    yobj(1) = min(yact);
    in_hyp = inithyp(X_trainset,y_trainset,meanfunc,covfunc,likfunc,nh,hypit);

    % Hyper parameter optimization function
    hyp2 = minimize(in_hyp, @gp, hypit, @infGaussLik, meanfunc, covfunc, likfunc, X_trainset,y_trainset);

    X_train = X_trainset;
    y_train = y_trainset;
    hyp1 = hyp2;
   
    for z = 1:q
 
        tic;
        dt = 0.1;
        fun2 = @(x) GU(X_train,y_train,hyp1,covfunc,x,meanfunc,likfunc,dt,z);
        [X_add,einew,~,~] = particleswarm(fun2,D,lb,ub);
        lt = toc;        
        y_add = func_eval(Eval_func,X_add,c);
        ytrue = func_eval(Eval_func,X_add,'NA');

        X_train = vertcat(X_train, X_add);
        y_train = vertcat(y_train, y_add);
        yact = vertcat(yact, ytrue);
        hyp1 = minimize(hyp1, @gp, hypit, @infGaussLik, meanfunc, covfunc, likfunc, X_train,y_train);
        ybest = min(yact);
        yobj(z+1) = ybest;
        res_time(z) = lt;


    end

    times(:,i) = res_time;
    yobjval(:,i) = yobj;
    
end

% Get Final Average MSE 
Final_time = mean(times,2);
Final_yobj = mean(yobjval,2);
alld = zeros((2*(q+1))+1,(req*3));
alld(q+4:(2*(q+1))+1,1:req) = times;
alld(2:q+1,1) = Final_time;
alld(q+4:(2*(q+1))+1,req+1:(2*req)) = sigvl;
alld(q+3:(2*(q+1))+1,(2*req)+1:(3*req)) = yobjval;
alld(1:q+1,2) = Final_yobj;

row_header1 = zeros(q+1,1);
row_header1(1:2*(q+1)+1,1) = (linspace(0,(2*q)+2,(2*q)+3))';

alld = [row_header1,alld];
for k = 1:3
    if k == 1
        headernew{k} = ['Num'];
    elseif k == 2
        headernew{k} = ['Avg_time'];
    elseif k == 3
        headernew{k} = ['Avg_Yobj'];
    end
end
     

cHeader = headernew;
commaHeader = [cHeader;repmat({','},1,numel(cHeader))]; %insert commaas
commaHeader = commaHeader(:)';
textHeader = cell2mat(commaHeader); %cHeader in text with commas
%write header to file
fid = fopen(filename,'w'); 
fprintf(fid,'%s\n',textHeader);
fclose(fid);
%write data to end of file
dlmwrite(filename,alld,'-append');
Gdata = Final_yobj;
Gtime = Final_time;
%%
filename = strcat(Eval_func,'_',num2str(in_tr),'_',num2str(req),'_',num2str(noise_level),'_','Mean','.csv');
na = 5;
Num = 0:q+1;
final = [Edata Sdata Kdata Gdata Bdata];
 

finaldata = zeros((q+1),na+1);
finaldata(1:q+1,1) = (linspace(0,q,q+1))';
finaldata(1:q+1,2:na+1) = final ;


cHeader = {'Num' 'EI' 'SKO' 'KG' 'GPUCB' 'BREI'}; %dummy header
commaHeader = [cHeader;repmat({','},1,numel(cHeader))]; %insert commaas
commaHeader = commaHeader(:)';
textHeader = cell2mat(commaHeader); %cHeader in text with commas
%write header to file
fid = fopen(filename,'w'); 
fprintf(fid,'%s\n',textHeader);
fclose(fid);
%write data to end of file
dlmwrite(filename,finaldata,'-append');
%%
filename = strcat(Eval_func,'_',num2str(in_tr),'_',num2str(req),'_',num2str(noise_level),'_','Meantime','.csv');
na = 5;
Num = 0:q+1;
final = [Etime Stime Ktime Gtime Btime];
 

finaldata = zeros((q+1),na+1);
finaldata(1:q+1,1) = (linspace(0,q,q+1))';
finaldata(2:q+1,2:na+1) = final ;


cHeader = {'Num' 'EI' 'SKO' 'KG' 'GPUCB' 'BREI'}; %dummy header
commaHeader = [cHeader;repmat({','},1,numel(cHeader))]; %insert commaas
commaHeader = commaHeader(:)';
textHeader = cell2mat(commaHeader); %cHeader in text with commas
%write header to file
fid = fopen(filename,'w'); 
fprintf(fid,'%s\n',textHeader);
fclose(fid);
%write data to end of file
dlmwrite(filename,finaldata,'-append');

