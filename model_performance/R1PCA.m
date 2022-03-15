function W = R1PCA(X, W0)

% X in d-N, W0 in d-K
NITER = 1000;
EPS = 1e-3;
[~, N] = size(X);
W = W0;
c = median(sum(X.^2) - sum((W0'*X).^2,1))^.5;
for iter = 1:NITER
    w = c * ones(1,N);
    w = w ./ sum((X - W*(W'*X)).^2);
    w(w > 1) = 1;
    C = X * (w .* X)';
    W_prev = W;
    W = C*W;
    W = orth(W);
    if norm(W_prev - W, 'fro') < EPS
        break;
    end
end

end