
clear all; clc;

%% set step-size parameter on each data set
%% a6a: ok,11,220*123
%% colon-cancer: alpha = 1e-10, beta = 1e-2, ok,62*2000
%% gisette: alpha = 1e-5, beta = 1e3,ok,6000*5000
%% ijcnn1: d is too small is applicable 49990*22,ok
% rcv1.binary: alpha = 1e-10, beta = 1e1
% real-sim: alpha = 1e-10, beta = 1
% w8a: alpha = 1e-8, beta = 1e-1

%% load real-world data set
% [y, X] = libsvmread('datasets\a6a.txt'); 
% [y, X] = libsvmread('datasets\colon-cancer'); 
% [y, X] = libsvmread('datasets\gisette_scale'); 
[y, X] = libsvmread('datasets\ijcnn1');
cluster_num = size(unique(y),1);
X = X'; [d, n] = size(X); 
% rng(1);
%% choose the dimension of subspace by the explained variance of PCA
p = min(n,d); 
if p < 10000
    [U,S,V] = svds(X,p); s = diag(S);
    for k = 1:p
        if sqrt(norm(s(1:k))^2/norm(s)^2) >= 0.8
            break;
        end
    end
    K = k;
else
    K = 50;
end

%% choose the running algorithm
run_PD = 1; run_PALMe = 1; run_PALM = 1; run_IP = 1; run_GS = 1;

%% set the parameters 
num_repeat = 40; 
accuracy_PE=zeros(1,num_repeat); accuracy_PA=zeros(1,num_repeat);
accuracy_IP=zeros(1,num_repeat); accuracy_GS=zeros(1,num_repeat);
accuracy_DC=zeros(1,num_repeat);

fval_PE = zeros(1,num_repeat);fval_PA = zeros(1,num_repeat);
fval_IP = zeros(1,num_repeat); fval_GS = zeros(1,num_repeat);
fval_DC = zeros(1,num_repeat);

time_PE = zeros(1,num_repeat); time_PA = zeros(1,num_repeat);
time_IP = zeros(1,num_repeat); time_GS = zeros(1,num_repeat);
time_DC = zeros(1,num_repeat);

for j = 1:num_repeat
     
    fprintf('Number of test: %d \n', j);
    F = randn(d,K); [Q0,~,~] = svd(F,'econ'); P0 = sign(randn(n,d));
    
    %% Proximal Alternating Linearized Minimization with extrapolation (PALMe)
    if run_PALMe == 1
        alpha_PE=1e-6; beta_PE = 1e4;
        tic;
        [Q_PE, P_PE,iter_2] = PALMe(X, Q0, P0, alpha_PE, beta_PE, 1);          
        time_PE(j) = toc;
        fval_PE(j) = sum(sum(abs(X'*Q_PE*Q_PE')));  
        accuracy_PE(j) = ClusteringMeasure(y,X'*Q_PE,cluster_num);
        fprintf('PALMe: accuracy = %f, critical gap = %f, time = %f, fval = %f\n',...
            accuracy_PE(j), norm(P_PE-sign(X'*Q_PE*Q_PE'),'fro'), time_PE(j), fval_PE(j));
    end
    
    %% Proximal Alternating Linearized Minimization (PALM)
    if run_PALM == 1
        alpha_PA =1e-7; beta_PA=1e5;
        tic;
        [Q_PA, P_PA,iter_1] = PALMe(X, Q0, P0, alpha_PA, beta_PA, 0);
        time_PA(j) = toc;
        fval_PA(j) = sum(sum(abs(X'*Q_PA)));   
        accuracy_PA(j) = ClusteringMeasure(y,X'*Q_PA,cluster_num);
        fprintf('PALM: accuracy = %f, critical gap = %f, time = %f, fval = %f\n',...
            accuracy_PA(j), norm(P_PA-sign(X'*Q_PA*Q_PA'),'fro'), time_PA(j), fval_PA(j));
    end
    
    %% Inertial Proximal Alternating Linearized Mimization (iPALM)
    if run_IP == 1
        alpha_IP =1e-9; beta_IP =1e5;
        tic;
        [Q_IP, P_IP,iter_4] = iPALM(X, Q0, P0, alpha_IP, beta_IP,1);
        time_IP(j) = toc;
        fval_IP(j) = sum(sum(abs(X'*Q_IP)));   
        accuracy_IP(j) = ClusteringMeasure(y,X'*Q_IP,cluster_num);
        fprintf('iPALM: accuracy = %f, critical gap = %f, time = %f, fval = %f\n',...
            accuracy_IP(j), norm(P_IP-sign(X'*Q_IP*Q_IP'),'fro'), time_IP(j), fval_IP(j));
    end

    %% Gauss-Seidel Inertial Proximal Alternating Linearized Mimization (GiPALM)
    if run_GS == 1
        alpha_GS = 1e-6; beta_GS=1e4;
        tic;
        [Q_GS, P_GS,iter_5] = GiPALM(X, Q0, P0, alpha_GS, beta_GS,1);
        time_GS(j) =toc;
        fval_GS(j) = sum(sum(abs(X'*Q_GS)));    
        accuracy_GS(j) =ClusteringMeasure(y,X'*Q_GS,cluster_num);
        fprintf('GiPALM: accuracy = %f, critical gap = %f, time = %f, fval = %f\n',...
            accuracy_GS(j), norm(P_GS-sign(X'*Q_GS*Q_GS'),'fro'), time_GS(j), fval_GS(j));
    end
    
    %% proximal DC with extrapolation (PDCe)
    if run_PD == 1 
        beta_PD = 1e-4;
        tic;
        [Q_DC, iter_3] = PDCe(X,Q0, beta_PD, 1);
        time_DC(j) = toc;
        fval_DC(j) = sum(sum(abs(X'*Q_DC))); 
        accuracy_DC(j) = ClusteringMeasure(y,X'*Q_DC,cluster_num);
        fprintf('PDCe: accuracy = %f, time = %f, fval = %f\n', accuracy_DC(j), time_DC(j), fval_DC(j));
    end    
end

%% record the information
fprintf('********** average accuracy and time of each method ********** \n')
if run_PALMe == 1 
    ave_accuracy_PE = sum(accuracy_PE) / num_repeat;
    ave_time_PE = sum(time_PE) / num_repeat;
    fprintf('PALMe: accuracy = %f, time = %f\n', ave_accuracy_PE, ave_time_PE);
end
if run_PALM == 1 
    ave_accuracy_PA = sum(accuracy_PA) / num_repeat;
    ave_time_PA = sum(time_PA) / num_repeat;
    fprintf('PALM: accuracy = %f, time = %f\n', ave_accuracy_PA, ave_time_PA);
end

if run_PD == 1 
    ave_accuracy_DC = sum(accuracy_DC) / num_repeat;
    ave_time_DC = sum(time_DC) / num_repeat;
    fprintf('PDCe: accuracy = %f, time = %f\n', ave_accuracy_DC, ave_time_DC);
end
if run_IP == 1 
    ave_accuracy_IP = sum(accuracy_IP) / num_repeat;
    ave_time_IP = sum(time_IP) / num_repeat;
    fprintf('iPALM: accuracy = %f, time = %f\n', ave_accuracy_IP, ave_time_IP);
end
if run_GS == 1 
    ave_accuracy_GS = sum(accuracy_GS) / num_repeat;
    ave_time_GS = sum(time_GS) / num_repeat;
    fprintf('GiPALM: accuracy = %f, time = %f\n', ave_accuracy_GS, ave_time_GS);
end

% fprintf("The accuracy: PALMe =%f,PALM = %f,pDCAe= %f,iPALM =%f,GiPALM =%f\n", accuracy_PE(j),accuracy_PA(j), accuracy_DC(j),accuracy_IP(j), accuracy_GS(j));
