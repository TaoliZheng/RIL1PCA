function [Q, P, iter] = PALMe_line_search(X, Q, P, alpha,beta,extra,rho)
    
    fprintf('Proximal Alternating Linearized Maximation with Extrapolation for L1-PCA \n');
    
    %% initial setting  
    X_QQ = X'*Q*Q'; X_E = X_QQ;
    Q_update = 0;
    tol =1e-6; iternum =1000;
    print =1;
    
    for iter = 1:iternum          
            
            X_Q_old = X_QQ ; 
                        
            %% update P
            P = alpha*P + X_E; P = sign(P);  
                  
            %% update Q
%             beta =2.5* norm(X_P+X_P','fro');
            Q_old = Q; Q_update_old = Q_update;
            [U, ~, V] = svd(beta*Q + X*(P*Q)+P'*(X'*Q), 'econ'); Q = U * V'; X_QQ = X'*Q*Q'; Q_update = norm(Q-Q_old)^2;
            obj_update = trace(P'*(X_QQ-X_Q_old))-beta/2 *(Q_update-Q_update_old)
            if obj_update<0
                beta = beta*rho;
                Q_old = Q; Q_update_old = Q_update;
                [U, ~, V] = svd(beta*Q + X*(P*Q)+P'*(X'*Q), 'econ'); Q = U * V'; X_QQ = X'*Q*Q'; Q_update = norm(Q-Q_old)^2;
                obj_update = trace(P'*(X_QQ-X_Q_old))-beta/2 *(Q_update-Q_update_old)
            end
                
            %% extrapolation scheme
            if extra == 1
                gamma = 1;
            else
                gamma = 0;
            end  
            
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