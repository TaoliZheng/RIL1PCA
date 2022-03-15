function W = L21(X, W0)

% X in d-N, W0 in d-m
[~, N] = size(X);

NITER = 1000;
W = W0;
EPS = 1e-3;

for iter = 1:NITER
    V = W'*X; 
    zero_idx = (sum(V.^2) == 0);
    nz_idx = ~zero_idx;
    V(:,zero_idx) = 0;
    V(:,nz_idx) = V(:,nz_idx) ./ sum(V(:,nz_idx).^2).^.5;
    M =  X*V';
    [u, ~, v] = svd(M, 'econ');
    W_prev = W;
    W = u*v';
    if norm(W_prev - W, 'fro') < EPS
        break;
    end
end

end