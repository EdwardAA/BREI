function [lb,ub] = func_bounds(Eval_func,dim)
switch Eval_func
    case '1d' 
        lb =  [2.7];
        ub = [7.5];
    case '1d2' 
        lb =  [0];
        ub = [1];
    case '1d3' 
        lb =  [0];
        ub = [1.2];
    case '1d4' 
        lb =  [-10];
        ub = [10];
    case '1d5' 
        lb =  [-10];
        ub = [10];
    case 'Alpine' 
        lb =  (-5.* ones(dim,1))';
        ub = (5.* ones(dim,1))';
    case 'Bohavchevsky' 
        lb =  (-2.* ones(dim,1))';
        ub = (2.* ones(dim,1))';
    case 'Branin'  
        lb =  [-5,0];
        ub = [10,15];
    case 'Cscendes' 
        lb =  (-1.* ones(dim,1))';
        ub = (1.* ones(dim,1))';
    case 'DP' 
        lb =  (0.* ones(dim,1))';
        ub = (1.* ones(dim,1))';
    case 'EggHolder' 
        lb =  (-512.* ones(dim,1))';
        ub = (512.* ones(dim,1))';
    case 'GW' 
        lb =  (-1.* ones(dim,1))';
        ub = (1.* ones(dim,1))';
    case 'Levy' 
        lb =  (-5.* ones(dim,1))';
        ub = (5.* ones(dim,1))';
    case 'Plateau' 
        lb =  (-5.* ones(dim,1))';
        ub = (5.* ones(dim,1))';
    case 'Qing' 
        lb =  (-5.* ones(dim,1))';
        ub = (5.* ones(dim,1))';
    case 'Quartic' 
         lb =  (-1.* ones(dim,1))';
         ub = (1.* ones(dim,1))';
    case 'Rastrigin' 
        lb =  (-1.* ones(dim,1))';
        ub = (1.* ones(dim,1))';
    case 'Rosenbrock' 
        lb =  (-1.* ones(dim,1))';
        ub = (1.* ones(dim,1))';
    case 'SDP' 
         lb =  (-1.* ones(dim,1))';
         ub = (1.* ones(dim,1))';
    case 'SHC'  
        lb =  [-3,-2];
        ub = [3,2];
    case 'SHC2'  
        lb =  [-1.6,-0.8];
        ub = [2.4,1.2];
    case 'STang' 
         lb =  (-5.* ones(dim,1))';
         ub = (5.* ones(dim,1))';
    case 'THC'  
        lb =  (-2.* ones(dim,1))';
        ub = (2.* ones(dim,1))';
end
        
        
        

        
