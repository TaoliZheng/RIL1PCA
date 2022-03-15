function [Q, P, fval_collect, Q_collect,iter, time_collect] = PALMe(X, Q, P, alpha,beta, opts)
    
    fprintf('Proximal Alternating Linearized Maximation with Extrapolation for L1-PCA \n');

    %% default parameter setting
    iternum = opts.iternum;  
    tol = opts.tol; 
    if isfield(opts,'print')
        print = opts.print;
    else
        print = 0;
    end
    if isfield(opts, 'extra')
        extra = opts.extra;
    else
        extra = 0;
    end
    
    %% initial setting
    residu_P = []; residu_Q = []; fval_collect = [];  
    fval = trace(X'*Q*Q'*P'); fval_collect(1) = fval; Q_collect(:,:,1) = Q;
    X_QQ = X'*Q*Q'; X_E = X_QQ;
    time_collect(1) = 0;
    
    for iter = 1:iternum          
            
            tic;
            X_Q_old = X_QQ ; fval_old = fval;
                        
            %% check the optimality for P
            if print == 1                                                
                P1 = P + X_E;
                P1 = sign(P1);               
                P_residu = norm(P-P1); residu_P(iter) = P_residu; 
            end
            
            %% update P
            P = alpha*P + X_E; P = sign(P);  
            
            %% check the optimality for Q
            if print == 1
                Q1 = Q + X*(P*Q)+P'*(X'*Q); [U1, ~, V1] = svd(Q1, 'econ'); Q1 = U1*V1';  
                Q_residu = norm(Q - Q1); residu_Q(iter) = Q_residu;          
            end
            
            %% update Q
%             beta =2.5* norm(X_P+X_P','fro');
            [U, ~, V] = svd(beta*Q + X*(P*Q)+P'*(X'*Q), 'econ'); Q = U * V'; 
            
            %% extrapolation scheme
            if extra == 1
                gamma = 1;
            else
                gamma = 0;
            end     
            X_QQ = X'*Q*Q'; X_E = (1+gamma)*X_QQ - gamma*X_Q_old; 
                        
            %% collect and print the iterate information
            fval = trace(X'*Q*Q'*P'); fval_collect(iter+1) = fval; Q_collect(:,:,iter+1) = Q;
                       
            if print == 1
                fprintf('Iternum: %d, Residual of P: %f, Residual of Q: %f, fval: %f\n',  iter, P_residu, Q_residu,fval); 
            end
            
            %% check the stopping criterion
            if abs(fval - fval_old) <= tol% Q_residu + P_residu < tol
                time_collect(iter+1) = toc + time_collect(iter);
                iter = iter + 1;
                break;
            end
            
            time_collect(iter+1) = toc + time_collect(iter);
    end
    
end