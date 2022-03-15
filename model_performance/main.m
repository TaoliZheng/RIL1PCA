clear; clc;

%% load real-world data set
% [y, X] = libsvmread('datasets\heart_scale.txt');
% [y, X] = libsvmread('datasets\wine.scale');  
% [y, X] = libsvmread('datasets\dna.scale'); 
% [y, X] = libsvmread('datasets\australian_scale.txt'); 
[y, X] = libsvmread('datasets\mushrooms.txt'); 
cluster_num = size(unique(y),1);
X = X'; [d, n] = size(X); 

%% choose the dimension of subspace by the explained variance of PCA
p = min(n,d); 
if p < 10000
    [U,S,V] = svds(X,p); s = diag(S);
    for k = 1:p
        if sqrt(norm(s(1:k))^2/norm(s)^2) >= 0.95
            break;
        end
    end
    K = k;
else
    K = 50;
end

%% center data
X = bsxfun(@minus,X,mean(X,2));

%% set the parameter
num_repeat = 1;  
%% compute the K leading vectors 
[Q,S] = eigs(X*X',K); var = sum(diag(S));

acc_1 = zeros(1,num_repeat); acc_2 = zeros(1,num_repeat); acc_3 = zeros(1,num_repeat);
acc_4 = zeros(1,num_repeat); acc_5 = zeros(1,num_repeat); acc_6 = zeros(1,num_repeat);

time_L21 = zeros(1,num_repeat); time_RI = zeros(1,num_repeat); time_RIL1 = zeros(1,num_repeat);
time_L1 = zeros(1,num_repeat); time_awl1pca = zeros(1,num_repeat); time_RL1_PCA = zeros(1,num_repeat);

for j = 1:num_repeat

    %% generate initial point: P0, Q0
    F = randn(d, K); [U,S,V] = svd(F,'econ'); Q0 = U(:,1:K);%d*K
    P0 = ones(n,d).*sign(randn(n,d)); %n*d
    P1 = ones(n,K).*sign(randn(n,K)); 
    
    %% L21-PCA 
    % Non-Greedy L21-Norm Maximization for Principal Component Analysis
    tic;
    Q_21 = L21(X, Q0);
    time_L21(j) =toc;
    acc_1(j) =  ClusteringMeasure(y,X'*Q_21,cluster_num);

    %% RIPCA
    %R1-PCA: Rotational invariant L1-norm principal component analysis for robust subspace factorization
    tic;
    Q_RI = R1PCA(X, Q0);
    time_RI(j) = toc;
    acc_2(j) =  ClusteringMeasure(y,X'*Q_RI,cluster_num);
        
    %% RI-L1PCA(Our maximization model)
    %% set the step-size parameter
    alpha_RL1 = 1e-8; beta_RL1 =1e4;
    % alpha = 1e-6; beta = 1e-1; heart
    % alpha = 1e-7; beta =1e2; wine
    % alpha = 1e-6; beta =1e2; dna
    % alpha = 1e-6; beta =5e4;australian
    tic; 
    [P_RIL1,Q_RIL1] = PALMe(X, Q0, P0, alpha_RL1, beta_RL1);
    time_RIL1(j)= toc;  
    acc_3(j) =  ClusteringMeasure(y,X'*Q_RIL1,cluster_num);
    fprintf("RI-L1PCA: The critical gap is %f\n",norm(P_RIL1-sign(X'*Q_RIL1*Q_RIL1'),'fro'));

    %% Ll-PCA
    %Linear Convergence of a Proximal Alternating Minimization Method with Extrapolation for ℓ1-Norm Principal Component Analysis
    alpha_L1 = 1e-8; beta_L1 = 1e3; 
    % alpha = 1e-3; beta = 1; heart
    % alpha = 1e-5; beta = 1e-1; glass
    % alpha = 1e-6; beta = 1e1; wine
    % alpha = 1e-5; beta = 1e-1; dna
    % alpha = 1e-5; beta = 1e1; australian
    % alpha = 1e-6; beta = 1e2; pendigits
    tic; 
    [P_L1,Q_L1] = PAMe(X, Q0, P1, alpha_L1, beta_L1);
    time_L1(j) = toc;      
    acc_4(j) = ClusteringMeasure(y,X'*Q_L1,cluster_num);
    fprintf("L1-PCA: The critical gap is %f\n",norm(P_L1-sign(X'*Q_L1),'fro'));
  
    %% awl1pca
    % Iteratively Reweighted Least Squares Algorithms for L1-Norm Principal Component Analysis
    tic;
    Q_awl1pca = aw1pca(X',K,n,Q0);
    time_awl1pca(j) = toc;
    acc_5(j) = ClusteringMeasure(y,X'*Q_awl1pca,cluster_num);

    %% RL1_PCA
    % An efficuent algorithm for l1-norm principal component analysis
    tic;
    [U,V]=RL1_PCA(X,K,d,n);
    time_RL1_PCA(j) = toc;
    acc_6(j) = ClusteringMeasure(y,X'*U,cluster_num);

end
ave_acc_1 = sum(acc_1)/num_repeat; ave_acc_2 = sum(acc_2)/num_repeat; ave_acc_3 = sum(acc_3)/num_repeat; 
ave_acc_4 = sum(acc_4)/num_repeat; ave_acc_5 = sum(acc_5)/num_repeat; ave_acc_6 = sum(acc_6)/num_repeat; 

ave_time_1 = sum(time_L21)/num_repeat; ave_time_2 = sum(time_RI)/num_repeat; ave_time_3 = sum(time_RIL1)/num_repeat;
ave_time_4 = sum(time_L1)/num_repeat; ave_time_5 = sum(time_awl1pca)/num_repeat; ave_time_6 = sum(time_RL1_PCA)/num_repeat;

fprintf("The average accuracy of model L21_PCA=%f,RIPCA=%f,RI-L1PCA=%f, Ll-PCA=%f,awl1pca=%f,RL1_PCA=%f\n",ave_acc_1,...
ave_acc_2,ave_acc_3,ave_acc_4,ave_acc_5,ave_acc_6);

fprintf("The average time of model L21_PCA=%f,RIPCA=%f,RI-L1PCA=%f, Ll-PCA=%f,awl1pca=%f,RL1_PCA=%f\n",ave_time_1,...
ave_time_2,ave_time_3,ave_time_4,ave_time_5,ave_time_6);
