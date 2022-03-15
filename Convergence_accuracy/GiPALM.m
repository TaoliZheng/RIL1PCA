function [Q, P,iter] = GiPALM(X, Q, P, alpha, beta, extra)
    %%%% implement the Guass-Seidel type inertial PALM in Gao et al. (2019) for L1-PCA %%%% 
    fprintf('********* Gauss-Seidel Inertial Proximal Alternating Linearized Mimization for L1-PCA *********\n');
    %% initial setting
    tol=1e-6;iternum =1000;
    X_QQ = X'*Q*Q'; E_P = P; E_Q = Q; X_E_Q = X_QQ;
    print =1;
    
    for iter = 1:iternum
            
            %% inertial scheme
            if extra == 1
                gamma_P = 1/2; gamma_Q = 1/4; 
            else
                gamma_P = 0; gamma_Q = 0;
            end    
          
            
            %% update P
            E_P_old = E_P; 
            P = alpha*E_P + X_E_Q; P = sign(P);  
            E_P = P + gamma_P * (P-E_P_old); 
            
            %% update Q
            E_Q_old = E_Q;
            [U, ~, V] = svd(beta*E_Q + X*(E_P*Q)+E_P'*(X'*Q), 'econ'); 
            Q = U * V';    
            
            %% extrapolation scheme
            E_Q = Q + gamma_Q * (Q - E_Q_old); 
            X_QQ=X'*Q*Q';
            X_E_Q = X' * E_Q * E_Q';
            
            if print==1
                fprintf('iter=%d,residual=%f\n',iter,norm(Q-E_Q_old,'fro'));
            end
            
            %% check the stopping criterion
            if norm(Q - E_Q_old,'fro') <= tol % Q_residu + P_residu
                iter = iter+1;
                break;
            end
    end
    
end