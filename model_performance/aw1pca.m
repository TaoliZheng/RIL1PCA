function [pc_best] = aw1pca(X,K,n,Q0)
%% initial setting
    tol = 1e-3;
    NITER = 1000;
    beta = 0.99;
    gamma = 0.1;
    pc_best = Q0;
    w_prev = 2*ones(n,1);
    w = ones(n,1);
    u = ones(n,1);
    max_u = 0;
    obj_best = Inf;
    
    for t=1:NITER
        X_new = diag(sqrt(w))*X;
        X_diff = X_new -X;
        %% update eigenvalues and eigenvectors
        if norm(X_diff)^2 < gamma *norm(X_new)^2 && t>1
            [ev,pc] = L2PCA_approx(ev,pc,K,X_diff'*X_diff);
        else
            [coff,~,latent] = pca(X_new);
            pc = coff(:,1:K);
            ev = latent(1:K);
        end
        %% set stopping critria
        if norm(pc-pc_best,'fro') < tol
            pc_best = pc;
            break;   
        end
        %% update function value
        Reconstruction_error = X - X*pc*pc';
        obj = sum(sum(abs(Reconstruction_error)));
        if obj < obj_best
            obj_best = obj;
            pc_best = pc;
        end
        %% update weights
        for i=1:n
            l2_norm_loss = sum(Reconstruction_error(i,:).^2);
            l1_norm_loss = sum(abs(Reconstruction_error(i,:)));
            if l2_norm_loss > 0.1
                u(i) = l1_norm_loss/l2_norm_loss;
                max_u = max(max_u,u(i));
            else
                u(i) = -1;
            end
            u(u(:,1)==-1)=max_u;
        end
        for i=1:n
            if u(i)< w(i)*(1-beta)
                w(i) = w(i)*(1-beta^t);
            elseif u(i)> w(i)*(1-beta)
                w(i) = w(i)*(1+beta^t);
            end
        end
    end
    
        
                
        
        
        
    
