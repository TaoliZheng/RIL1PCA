function [Q,iter] = PDCe(X, Q, beta, extra)
    fprintf('********* Proximal DC with extrapolation for L1-PCA *********\n');
    %% Default parameter setting
    tol = 1e-6;iternum =1000;
    
    %% initial setting
    theta = 1; %% extrapolation stepsize
    Q_extra = Q; X_QQ = X'*Q*Q'; subg = X*(sign(X_QQ)*Q)+sign(X_QQ')*(X'*Q);
    print=1;

    for iter = 1:iternum
        
        %% proximal DC step
        Q_old = Q; theta_old = theta;              
        [U, ~, V] = svd(Q_extra + beta*subg, 'econ'); Q = U * V';   

        %% fixed restarting scheme
        if extra == 1
            if mod(iternum, 10) == 1 
                theta = 1; theta_old = theta;
            else
                theta = 0.5*(1+sqrt(1+4*theta^2)); 
            end     
            gamma = (theta_old - 1)/theta;
        else
            gamma = 0;
        end
        
        %% extrapolation update        
        Q_extra = Q + gamma * (Q - Q_old);
        X_QQ = X'*Q*Q';
        fval = sum(sum(abs(X_QQ))); 

        %% update
        subg = X*(sign(X_QQ)*Q)+sign(X_QQ')*(X'*Q);
        
        if print==1
                fprintf('iter=%d,residual=%f\n',iter,norm(Q - Q_old,'fro'));
        end
        
        %% check the stopping criterion
        if norm(Q - Q_old,'fro') < tol
            iter = iter+1;
            break;
        end
    end
end



