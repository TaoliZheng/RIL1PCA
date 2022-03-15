
function [P,Q] = PAMe(X, Q, P, alpha, beta) 
    tol = 1e-3;
    gamma = 1;
    NITER = 1000;
    
    %% initial setting
    X_Q = X'*Q; X_E = X_Q;
    
    for iter = 1:NITER   
            
            X_Q_old = X_Q;
            
            %% update P
            P = alpha*P + X_E;
            P = sign(P);
           
            %% update Q
            [U, ~, V] = svd(beta*Q + X*P, 'econ'); Q = U * V'; 
            
            %% extrapolation scheme
            X_Q = X'*Q;
            X_E = (1+gamma)*X_Q - gamma*X_Q_old; 
            
            fprintf("The iter is %d, residual is %f\n",iter,norm(X_Q-X_Q_old,'fro'));
            %% check the stopping criterion
            if norm(X_Q-X_Q_old,'fro') < tol
                break;
            end
    end    
end



