%*******************  EE556 - Mathematics of Data  ************************
% Function:  [x, info] = ADAM(fx, gradf, parameter)       
% Purpose:   Implementation of the AdaGrad algorithm.     
% Parameter: x0         - Initial estimate.
%            maxit      - Maximum number of iterations.
%*************************** LIONS@EPFL ***********************************

function [x,info] = ADAM(fx, gradf, parameter)

    fprintf('%s\n', repmat('*', 1, 68));
    fprintf('ADAM\n')

    % Initialization
    %%%% YOUR CODES HERE
	    
    % Main loop.
    for iter    = 1:parameter.maxit
            
        % Start timer
        tic;
       
        % Update the next iteration. (main algorithmic steps here!)
        % Use the notation x_next for x_{k+1}, and x for x_{k}, and similar for other variables.              
        %%%% YOUR CODES HERE
        
       
        % Compute error and save data to be plotted later on.
        info.itertime(iter ,1)      = toc;
        info.fx(iter, 1)            = fx(x); % fx(x_bar/iter);
        
         % Print the information.
        if mod(iter, 5) == 0 || iter == 1
        fprintf('Iter = %4d,  f(x) = %0.9f\n', ...
                iter,  info.fx(iter, 1));
        end
        
        % Prepare the next iteration
        x   = x_next;

    end

    % Finalization.
    info.iter           = iter;
end

