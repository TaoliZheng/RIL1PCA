function [U,V]=RL1_PCA(X,K,d,n)
%%initial setting
mu = 1/norm(X,'fro');
rou = 1.2;
E = zeros(d,n);
A = zeros(d,n);
NITER=1000;
tol = 1e-3;
for i=1:NITER
    %% update U and V
    E_old = E;
    [U, Sigma, V] = svds(X-E-A/mu,K); 
    V = Sigma*V';
    %% update E
    P = X-U*V+A/mu;
    E = sign(P).*max(abs(P)-1/mu,0);
    %% update A
    A = A+mu*(X-U*V-E);
    mu = min(mu*rou,1e11);
    %% stopping critieria
    if norm(E-E_old,'fro')< tol
        break;
    end
end
    
    
     
