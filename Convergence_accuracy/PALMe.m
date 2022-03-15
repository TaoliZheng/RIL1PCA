function [Q, P, iter] = PALMe(X, Q, P, alpha,beta,extra)
    
    fprintf('Proximal Alternating Linearized Maximation with Extrapolation for L1-PCA \n');
    
    %% initial setting  
    X_QQ = X'*Q*Q'; X_E = X_QQ;
    tol =1e-6; iternum =1000;
    print =1;
    
    for iter = 1:iternum          
            
            X_Q_old = X_QQ ; 
                        
            %% update P
            P = alpha*P + X_E; P = sign(P);  
                  
            %% update Q
%             beta =2.5* norm(X_P+X_P','fro');
            [U, ~, V] = svd(beta*Q + X*(P*Q)+P'*(X'*Q), 'econ'); Q = U * V'; 
            
            %% extrapolation scheme
            if extra == 1
                gamma = 1;
            else
                gamma = 0;
            end  
            
            X_QQ = X'*Q*Q';
            X_E = (1+gamma)*X_QQ - gamma*X_Q_old; 
            
            if print==1
                fprintf('iter=%d,residual=%f\n',iter,norm(X_E-X_Q_old,'fro'));
            end
            
            %% check the stopping criterion
            if norm(X_QQ-X_Q_old,'fro') <= tol
                iter = iter + 1;
                break;
            end
    end
    
end