function y = func_eval(Eval_func,x,cval)

switch Eval_func
    case '1d' 
        y = sin(x)+sin((10/3).*x);
    case '1d2' 
        y = ((6*x-2).^2).*sin((12*x)-4);
    case '1d3' 
        y = (-(1.4-3*x).*sin(18*x));
    case '1d4' 
        y = (-(x+sin(x))).*(exp(-(x.^2)));
    case '1d5' 
        y = zeros(size(x,1),1);
        for i = 1:size(x,1)
            sum1 = 0;
            for j = 1:6
                sum1 = -(j*sin((j+1)*x(i,1)+j))+sum1;
            end
            y(i,:) = sum1;
        end
    case 'Alpine' 
        y = zeros(size(x,1),1);
        for i = 1:size(x,1)
            sum1 = 0;
            for j = 1:size(x,2)
                sum1 = abs(x(i,j)*sin(x(i,j)+0.1*x(i,j)))+sum1;
            end
            y(i,:) = sum1;
        end
    case 'Bohavchevsky' 
        y = zeros(size(x,1),1);
        for i = 1:size(x,1)
            sum1 = 0;
            for j = 1:(size(x,2))-1
                sum1 = (x(i,j)^2)+2*(x(i,j+1)^2)-0.3*(cos(3*pi*x(i,j)))-0.4*(cos(4*pi*x(i,j+1)))+0.7+sum1;
            end
            y(i,:) = sum1;
        end
    case 'Branin'  
        x1 = x(:,1);
        x2 = x(:,2);
        y = (x2-(5.1/(4*pi^2))*(x1.^2)+((5/pi)*x1)-6).^2+10*(1-(1/(8*pi)))*cos(x1)+10;

    case 'Cscendes' 
        y = zeros(size(x,1),1);
        for i = 1:size(x,1)
            sum1 = 0;
            for j = 1:size(x,2)
                sum1 = (x(i,j)^6)*(2+sin(1/x(i,j)))+sum1;
            end
            y(i,:) = sum1;
        end
    case 'DP'  
        x1 = x(:,1);
        x2 = x(:,2);
        x3 = x(:,3);
        e1 = (x1-2+8*x2-8*x2.^2).^2;
        e2 = (3-4*x2).^2;
        e3 = 16*(sqrt(x3+1).*((2*x3-1).^2));
        e = e1+e2+e3;
        y = 4*e;
    case 'EggHolder'  
        x1 = x(:,1);
        x2 = x(:,2);
        y = -(x2+47).*sin(sqrt(abs(x2+(x1./2)+47)))-x1.*sin(sqrt(abs(x1-(x2+47))));
    case 'GW' 
        y = zeros(size(x,1),1);
        for i = 1:size(x,1)
            sum1 = 0;
            for j = 1:size(x,2)
                sum1 = ((x(i,j)^2)/4000)+sum1;
            end
            for k = 1:size(x,2)
                prod1 = 1;
                prod1 = prod1*cos(x(i,j))/sqrt(k);
            end
            y(i,:) = sum1-prod1+1;
        end
    case 'Levy' 
        y = zeros(size(x,1),1);
        x1 = x(:,1);
        w1 = 1+(x1-1)./4;
        term1 = sin(pi*w1).^2;
        k = size(x,2);
        for i = 1:size(x,1)
            sum1 = 0;
            for j = 1:(k-1)
                sum1 = (1+(x(i,j)-1)/4)^2*(1+10*sin(pi*(1+(x(i,j)-1)/4)+1)^2)+((1+(x(i,k)-1)/4)-1)^2*(1+sin(2*pi*(1+(x(i,k)-1)/4))^2)+sum1;
            end
            y(i,:) = term1(i) + sum1;
        end
     case 'Plateau' 
         y = zeros(size(x,1),1);
         for i = 1:size(x,1)
             sum1 = 0;
             for j = 1:size(x,2)
                sum1 = floor(x(i,j))+sum1;
             end
             y(i,:) = sum1;
         end
    case 'Qing'
         y = zeros(size(x,1),1);
         for i = 1:size(x,1)
             sum1 = 0;
             for j = 1:size(x,2)
                 sum1 = ((x(i,j)^2)-(j))^2+sum1;
             end
             y(i,:) = sum1;
         end
     case 'Quartic' 
         y = zeros(size(x,1),1);
         for i = 1:size(x,1)
             sum1 = 0;
             for j = 1:size(x,2)
                 sum1 = (j*(x(i,j)^4))+sum1;
             end
             y(i,:) = sum1;
         end
    case 'Rastrigin'
         y = zeros(size(x,1),1);
         for i = 1:size(x,1)
             sum1 = 0;
             for j = 1:size(x,2)
                 sum1 = (x(i,j)^2)-10*(cos(2*pi*x(i,j)))+sum1;
             end
             y(i,:) = 10*(size(x,2))+sum1;
         end
    case 'Rosenbrock'
         y = zeros(size(x,1),1);
         for i = 1:(size(x,1)-1)
             sum1 = 0;
             for j = 1:(size(x,2))-1
                 sum1 = (100*((x(i,j)^2)-x(i,j+1)^2))+(x(i,j)-1)^2+sum1;
             end
             y(i,:) = sum1;
         end
    case 'SDP'
         y = zeros(size(x,1),1);
         for i = 1:size(x,1)
             sum1 = 0;
             for j = 1:size(x,2)
                 sum1 = (abs(x(i,j))^(j+2))+sum1;
             end
             y(i,:) = sum1;
         end
    case 'STang'
         y = zeros(size(x,1),1);
         for i = 1:size(x,1)
             sum1 = 0;
             for j = 1:size(x,2)
                 sum1 = (x(i,j)^4)-16*(x(i,j)^2)+5*(x(i,j))+sum1;
             end
             y(i,:) = 0.5*sum1;
         end
    case 'THC'  
        x1 = x(:,1);
        x2 = x(:,2);
        y = 2*(x1.^2)-1.05*(x1.^4)+((x1.^6)/6)+x1.*x2+(x2.^2);
    case 'SHC'  
        x1 = x(:,1);
        x2 = x(:,2);
        y = 4*(x1.^2)+(x1.*x2)-4*(x2.^2)-2.1*(x1.^4)+4*(x2.^4)+(1/3)*(x1.^6);
     case 'SHC2'  
        x1 = x(:,1);
        x2 = x(:,2);
        y = 4*(x1.^2)+(x1.*x2)-4*(x2.^2)-2.1*(x1.^4)+4*(x2.^4)+(1/3)*(x1.^6);
end
        
if cval ~= 'NA'
    errtest = cval*randn(size(x,1),1);
    y = y + errtest;
end
end
        
        
        
        
