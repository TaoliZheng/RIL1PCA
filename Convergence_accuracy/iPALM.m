function [Q, P,iter] = iPALM(X, Q, P, alpha, beta, extra)
    
    %%%% implement the inertial PALM in Pock and Sabach (2016) for L1-PCA %%%% 
    fprintf('********* Inertial Proximal Alternating Linearized Mimization for L1-PCA *********\n');
    tol = 1e-6;iternum =1000;
    %% initial setting
    X_QQ = X'*Q*Q'; Q_old = Q; P_old = P;
    print=1;
    for iter = 1:iternum
            
            %% inertial scheme
            if extra == 1
                gamma_P = (iter-1)/(iter+2); gamma_Q = gamma_P; 
            else
                gamma_P = 0; gamma_Q = 0;
            end    
                       
            %% update P
            E_P = P + gamma_P * (P-P_old); P_old = P; 
            P = alpha*E_P + X_QQ; P = sign(P);  
            
            %% update Q
            E_Q = Q + gamma_Q * (Q - Q_old); Q_old = Q;
            [U, ~, V] = svd(beta*E_Q + X*(P*E_Q)+P'*(X'*E_Q), 'econ'); Q = U * V';         
            X_QQ = X'*Q*Q';
           
            if print==1
                fprintf('iter=%d,residual=%f\n',iter,norm(Q - Q_old,'fro'));
            end
            %% check the stopping criterion
            if norm(Q - Q_old,'fro') <= tol % Q_residu + P_residu < tol
                iter = iter + 1;
                break;
            enD
    end
    
end



