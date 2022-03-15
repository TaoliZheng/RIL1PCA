function [P,Q] = PALMe(X, Q, P, alpha,beta)
    
    tol = 1e-3;
    gamma = 1;
    NITER = 1000;
    %% initial setting
    X_QQ = (X'*Q)*Q'; X_E = X_QQ;
    
    for iter = 1:NITER          
            
            X_Q_old = X_QQ ;
                       
            %% update P
            P = alpha*P + X_E;  
            P = sign(P);
            
            %% update Q
%             beta =2.5* norm(X_P+X_P','fro'); 
            M = beta*Q + X*(P*Q)+P'*(X'*Q);
            [U, ~, V] = svd(M, 'econ');
            Q = U * V'; 
            
            %% extrapolation scheme  
            X_QQ = X'*Q*Q';
            X_E = (1+gamma)*X_QQ - gamma*X_Q_old; 
            
            fprintf("The iter is %d, residual is %f\n",iter,norm(X_E-X_Q_old, 'fro'));
            %% check the stopping criterion
            if norm(X_E-X_Q_old, 'fro') < tol
               break;
           end
    end
    
end