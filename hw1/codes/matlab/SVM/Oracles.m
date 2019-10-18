%*******************  EE556 - Mathematics of Data  ************************
% This function returns:
%  fx: objective function, i.e., fx(x) evaluate value of f at x
%  gradf: gradient mapping, i.e., gradf(x) evaluates gradient of f at x
%  gradfsto: stochastic gradient mapping, i.e., gradfsto(x,i) evaluates
%               gradient of f_i at x
%  hessf: Hessian mapping, i.e., hessf(x) evaluates hessian of f at x
%*************************** LIONS@EPFL ***********************************

function [ fx, gradf, gradfsto, hessfx ] = Oracles( b, A, sigma)

    function fx = fsmoothedhinge(A, b, x)
       
        n = size(A,1);
        fx = zeros(n,1);
        
        for i = 1:n
           
            yf = b(i) * A(i,:) * x;
            fx(i) = ( 0 < yf && yf <= 1 ) * (( 1 - yf )^2 / 2 ) + ( yf <= 0 ) * ( 0.5 - yf );
            
        end
        
        fx = mean(fx);
        
    end


    function gradfx = gradfsmoothedhinge(A, b, x)
       
        n = size(A,1);
        gradfx = zeros(size(x));
        for i = 1:n
           
            yf = b(i) * A(i,:) * x;
            if 0 < yf && yf <= 1
                gradfx = gradfx + ( ( yf - 1) ) * b(i) * A(i,:)';
            elseif yf <= 0
                gradfx = gradfx - b(i) * A(i,:)';
            end
            
        end

        gradfx = gradfx / n;
            
    end

    
    function hessfx = hessfsmoothedhinge(A, b, x)
    
        [n, p] = size(A);
        hessfx = zeros(p, p);
        
        for i = 1:n
            yf = b(i) * A(i,:) * x;
            if 0 < yf && yf <= 1
                hessfx = hessfx + A(i,:)' * A(i,:);
            end
            
        end
        
        hessfx = hessfx / n;
    
    end

    
    function stogradfx = stogradfsmoothedhinge(A, b, x, i)
    
        yf = b(i) * A(i,:) * x;
        
        
        if 0 < yf && yf <= 1
            stogradfx = (yf - 1) * b(i) * A(i,:)';
        elseif yf <= 0
            stogradfx = -1 * b(i) * A(i,:)';
        else
            stogradfx = zeros(size(x));
        end
        
    end


    [n, p] = size(A);
    
    fx      = @(x)(0.5*sigma*norm(x)^2 + fsmoothedhinge(A, b, x));
    gradf  = @(x)(sigma*x + gradfsmoothedhinge(A, b, x));
    gradfsto  = @(x, i)(sigma*x + stogradfsmoothedhinge(A, b, x, i));
    hessfx  = @(x)(sigma*eye(p) + hessfsmoothedhinge(A, b, x));

end

%**************************************************************************
% END OF THE IMPLEMENTATION.
%**************************************************************************
