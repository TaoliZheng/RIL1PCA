
clear; clc;

%% basic setting
n = 4000; %%% n = number of samples
d = 2000; %%% d = number of features
K = 50;   %%% K = dimension of subspace
rng(1);
%% generate the synthetic data using the l1-fixed effect model
Y = randn(d,K); U = Y*(Y'*Y)^(-0.5);
A = rand(K,n); A = A - mean(A,2);
X = U*A + laprnd(d,n,0,0.5);%d*n

%% choose the running algorithm
run_PALM = 1; run_PALMe = 1; run_iPALM =1; run_GS = 1; run_DC = 1; 

%% set the parameters
numinit = 1;  maxiter = 1e3; tol = 1e-8; print =1; extra = 1; 

%% compute the K leading vectors 
[Q,S] = eigs(X'*X,K); var = sum(diag(S));

for j = 1:numinit

    %% generate initial point: P0, Q0
    F = randn(d, K); [U,S,V] = svd(F,'econ'); Q0 = U(:,1:K);%d*K
    P0 = ones(n,d).*sign(randn(n,d)); %n*d
    
    %% set the step-size parameter
    alpha = 1e-6;
    
    %% Proximal Alternating Linearized Mimization with extrapolation (PAMe)
    if run_PALMe == 1
        beta = 1e-2; % beta=1e1,0.969746 (46)(2000,4000);
        %(4000,2000) 0.976963,55 (beta=1e1); 0.976982,59 (beta=1e-1); 0.977057,55 (beta=1e-3) 
        % 0.9770057,56 (beta=1e-5), ,0.976244,58(beta=1e3);
        % 0.977100(beta=1e-2)
        opts = struct('iternum', maxiter, 'tol', tol, 'print', print, 'extra', extra);
        [Q_PE, P_PE, fval_collect_PE, Q_collect_PE,iter_2,time_PE] = PALMe(X, Q0, P0, alpha, beta, opts); 
%         [Q_PE, P_PE, fval_collect_PE, Q_collect_PE,iter_2,time_PE] = PALMe_line_search(X, Q0, P0, alpha, beta, opts, 0.1);
        optval_PE(j) = sum(sum(abs(X'*Q_PE*Q_PE')));             
        fprintf('PALMe: fval of = %f, explained variance: %f, critical gap = %f\n',...
            optval_PE(j), norm(X'*Q_PE,'fro')^2/var, norm(P_PE-sign(X'*Q_PE*Q_PE'),'fro'));
    end
    
    %% Standard Proximal Linearized Alternating Mimization (PAM)
    if run_PALM == 1 
        beta = 1e1; % beta= 1e1,0.961137 (795) (2000,4000)
        % (4000,2000) beta= 1e1,
        opts = struct('iternum', maxiter, 'tol', tol, 'print', print, 'extra', 0);
        [Q_PA, P_PA, fval_collect_PA, Q_collect_PA,iter_1,time_PA] = PALMe(X, Q0, P0, alpha, beta, opts);
        optval_PA(j) = sum(sum(abs(X'*Q_PA)));   
        fprintf('PALM: fval = %f, explained variance: %f, critical gap = %f\n',...
            optval_PA(j), norm(X'*Q_PA,'fro')^2/var, norm(P_PA-sign(X'*Q_PA*Q_PA'),'fro'));
    end
            
    %% Inertial Proximal Alternating Linearied Mimization (iPALM)
    if run_iPALM == 1
        beta = 1e-4;% beta=1e-1 (669),(2000,4000),beta=1e1 is not ok.
        % (4000,2000)(0.971395,879)alpha =1e-6; beta=1e-4,1e-6;
        opts = struct('iternum', maxiter, 'tol', tol, 'print', print, 'extra', extra);
        [Q_IP, P_IP, fval_collect_IP, Q_collect_IP,iter_4,time_IP] = iPALM(X, Q0, P0, alpha, beta, opts);
        optval_IP(j) = sum(sum(abs(X'*Q_IP)));   
        fprintf('iPALM: fval of = %f, explained variance: %f, critical gap = %f\n',...
            optval_IP(j), norm(X'*Q_IP,'fro')^2/var, norm(P_IP-sign(X'*Q_IP*Q_IP'),'fro'));
    end
    
    %% Gauss-Seidel Inertial Proximal Alternating Linearied Mimization (GiPALM)
    if run_GS == 1
        beta = 1e3; %beta=1, 0.962328 (165),(2000,4000)
        %(4000,2000)alpha=1e-6, beta=1e3, (0.971855,298)
        opts = struct('iternum', maxiter, 'tol', tol, 'print', print, 'extra', extra);
        [Q_GS, P_GS, fval_collect_GS, Q_collect_GS,iter_5,time_GS] = GiPALM(X, Q0, P0, alpha, beta, opts);
        optval_GS(j) = sum(sum(abs(X'*Q_GS)));   
        fprintf('GiPALM: fval = %f, explained variance: %f, critical gap = %f\n',...
            optval_GS(j), norm(X'*Q_GS,'fro')^2/var, norm(P_GS-sign(X'*Q_GS*Q_GS'),'fro'));
    end
    
    %% Proximal Difference-of-Convex Algorithm with extrapolation (pDCAe)
    if run_DC == 1
        beta =1e4; %beta=1e2,0.961396 (781),(2000,4000)
        %(4000,2000) beta=1e2 is not ok; beta=1e4(799,0.971550)
        opts = struct('iternum', maxiter, 'tol', tol, 'print', print, 'extra', extra);
        [Q_DC, fval_collect_DC, Q_collect_DC, iter_3,time_DC] = PDCe(X,Q0, beta, opts);
        optval_DC (j)= sum(sum(abs(X'*Q_DC)));             
        fprintf('PDCE: fval of = %f, explained variance: %f\n', optval_DC(j), norm(X'*Q_DC,'fro')^2/var);
    end
end
fprintf('Explained Variance: PALMe = %f, PALM = %f, iPALM = %f, GiPALM = %f PDCAe =%f \n', ...
    norm(X'*Q_PE,'fro')^2/var, norm(X'*Q_PA,'fro')^2/var, ...
    norm(X'*Q_IP,'fro')^2/var, norm(X'*Q_GS,'fro')^2/var, norm(X'*Q_DC,'fro')^2/var);
%Explained Variance: PALMe = 0.969746, PALM = 0.961137, iPALM = 0.961109,
%GiPALM = 0.962328, pDCAe =  0.961396(2000,4000)
%Explained Variance: PALMe = 0.977100, PALM = 0.971268, iPALM = 0.971395,
%GiPALM = 0.971855 pDCAe = 0.971550(4000,2000)
%% plot the figures of convergence rate in terms of the iterate Qk       
color1 = [0, 0.4470, 0.7410]; color2 = [0.8500, 0.3250, 0.0980];
color3 = [0.9290 0.6940 0.1250]; color4 = [0.4940 0.1840 0.5560];
color5 = [0.4660 0.6740 0.1880]; color6 = [0.6350 0.0780 0.1840];

figure();
if run_PALM == 1
   Q_dist = zeros(iter_1,1);
    for i = 1:iter_1
       Q_dist(i) = norm(Q_collect_PA(:,:,i) - Q_collect_PA(:,:,iter_1), 'fro') + tol; 
    end
    semilogy(time_PA,Q_dist, '-s', 'Color', color1, 'LineWidth', 2); hold on;
end
if run_PALMe == 1
    Q_dist = zeros(iter_2,1);
    for i = 1:iter_2
       Q_dist(i) = norm(Q_collect_PE(:,:,i) - Q_collect_PE(:,:,iter_2), 'fro') + tol; 
    end
    semilogy(time_PE,Q_dist, '-o', 'Color', color2, 'LineWidth', 2); hold on;
end
if run_DC == 1
   Q_dist = zeros(iter_3,1);
    for i = 1:iter_3
       Q_dist(i) = norm(Q_collect_DC(:,:,i) - Q_collect_DC(:,:,iter_3), 'fro') + tol; 
    end
    semilogy(time_DC,Q_dist, '-<', 'Color', color6, 'LineWidth', 2); hold on;
end
if run_iPALM == 1
    Q_dist = zeros(iter_4,1);
    for i = 1:iter_4
       Q_dist(i) = norm(Q_collect_IP(:,:,i) - Q_collect_IP(:,:,iter_4), 'fro') + tol; 
    end
    semilogy(time_IP, Q_dist, '-d', 'Color', color3, 'LineWidth', 2); hold on;
end
if run_GS == 1
    Q_dist = zeros(iter_5,1);
    for i = 1:iter_5
       Q_dist(i) = norm(Q_collect_GS(:,:,i) - Q_collect_GS(:,:,iter_5), 'fro') + tol; 
    end
    semilogy(time_GS,Q_dist, '->', 'Color', color4, 'LineWidth', 2); hold on;
end

legend('PALM', 'PALMe','PDCAe', 'iPALM', 'GiPALM', 'FontSize', 11);
xlabel('Time', 'FontSize', 13); 
ylabel('$||\mathbf{Q}^\texttt{k}-\mathbf{Q}^\mathbf{*}||_\mathbf{F}$', 'Interpreter', 'latex', 'FontSize', 13);
xrange = max([time_PA(end),time_PE(end),time_DC(end),time_IP(end),time_GS(end)]);
xlim([0 xrange+100]);
% grid on
% box on
set(gca, 'GridAlpha',0.2);
set(gca, 'MinorGridAlpha',0.2);
set(gca,'linewidth',2);
set(gca,'FontSize',10);
set(gca, 'FontName', 'AGaramondPro-Regular')
set(gcf,'color','w');
ax = gca;
ax.XAxis.TickLabelFormat = '%.0f';
axes.SortMethod='ChildOrder';
hold off;
export_fig('convergence rate in terms of the iterate Qk 4000_2000.pdf');

%% plot the figures of convergence rate in terms of function value
figure(); tol = 1e-8;
if run_PALM == 1
    minval_PA = min(-fval_collect_PA); fval_collect = -fval_collect_PA - minval_PA + tol; 
    semilogy(time_PA,fval_collect, '-s', 'Color', color1, 'LineWidth', 2); hold on;
end
if run_PALMe == 1
    minval_PE = min(-fval_collect_PE); fval_collect = -fval_collect_PE - minval_PE + tol; 
    semilogy(time_PE,fval_collect, '-o', 'Color', color2, 'LineWidth', 2); hold on;
end
if run_DC == 1
    minval_DC = min(-fval_collect_DC); fval_collect = -fval_collect_DC - minval_DC + tol; 
    semilogy(time_DC,fval_collect, '-<', 'Color', color6, 'LineWidth', 2); hold on;
end
if run_iPALM == 1
    minval_IP = min(-fval_collect_IP); fval_collect = -fval_collect_IP - minval_IP + tol; 
    semilogy(time_IP,fval_collect, '-d', 'Color', color3, 'LineWidth', 2); hold on;
end
if run_GS == 1
    minval_GS = min(-fval_collect_GS); fval_collect = -fval_collect_GS - minval_GS + tol; 
    semilogy(time_GS,fval_collect, '->', 'Color', color4, 'LineWidth', 2); hold on;
end

legend('PALM', 'PALMe','PDCAe','iPALM', 'GiPALM', 'FontSize', 11); 
xlabel('Time', 'FontSize', 13);  
ylabel('$\texttt{h}(\mathbf{P}^\texttt{k},\mathbf{Q}^\texttt{k})-\texttt{h}(\mathbf{P}^*,\mathbf{Q}^*)$',...
    'Interpreter', 'latex', 'FontSize', 13);
xlim([0 xrange+100]);
% grid on
% box on
set(gca, 'GridAlpha',0.2);
set(gca, 'MinorGridAlpha',0.2);
set(gca,'linewidth',2);
set(gca,'FontSize',10);
set(gca, 'FontName', 'AGaramondPro-Regular')
set(gcf,'color','w');
ax = gca;
ax.XAxis.TickLabelFormat = '%.0f';
axes.SortMethod='ChildOrder';
hold off;
export_fig('convergence rate in terms of function value 4000_2000.pdf');
save("4000_2000_data");