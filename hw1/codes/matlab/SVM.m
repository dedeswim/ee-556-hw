%**************************************************************************
%*************************** LIONS@EPFL ***********************************
%**************************************************************************
clearvars
addpath('SVM/')
data = load('dataset/breast-cancer.mat');
A = data.features_train;
b = data.labels_train;
A_test = data.features_test;
b_test = data.labels_test;

% fs_opt = 0.045102342; % square hinge
% fs_opt = 0.037003410; % modified huber (h=0.5)
fs_opt = 0.039897199; % smoothed hinge loss


[n, p] = size(A);

fprintf('%s\n', repmat('*', 1, 68));
fprintf('Linear Support Vector Machine:\n')
fprintf('Squared Hinge Loss + Ridge regularizer\n')
fprintf('dataset : %s : %d x %d\n',  'breast-cancer', size(A,1), size(A,2))
fprintf('%s\n', repmat('*', 1, 68));

% Choose the solvers you want to call
solve.GD        = 1;
solve.GDstr     = 1;
solve.AGD       = 1;
solve.AGDstr    = 1;
solve.LSGD      = 1;
solve.LSAGD     = 1;
solve.AGDR      = 1;
solve.LSAGDR    = 1;
solve.AdaGrad   = 1;
solve.ADAM      = 1;
solve.SGD       = 1;
solve.SAG       = 1;
solve.SVR       = 1;



% Set parameters and solve numerically with GD, AGD, AGDR, LSGD, LSAGD, LSAGDR.
fprintf(strcat('Numerical solution process is started: \n'));

sigma             = 1e-4;
parameter.Lips    = norm(full(A))*norm(full(A'))/n + sigma;
parameter.strcnvx = sigma;
parameter.x0      = zeros(p, 1);
parameter.Lmax    = 0;

for i=1:n
    parameter.Lmax = max(norm(A(i,:))* norm(A(i,:)), parameter.Lmax);
end
parameter.Lmax = parameter.Lmax + sigma;


%%% FIRST AND SECOND ORDER METHODS
[fx, gradf, ~, hessf] = Oracles(b, A, sigma);

%% Call the solvers

parameter.maxit                 = 4000;             
if solve.GD
[x.GD     , info.GD        ]    = GD     (fx, gradf, parameter);
e.GD = compute_error(A_test,b_test,x.GD);
fprintf('Error w.r.t 0-1 loss: %1.3e\n',e.GD);
end

parameter.maxit                 = 4000;   
if solve.GDstr
[x.GDstr  , info.GDstr     ]    = GDstr  (fx, gradf, parameter);
e.GDstr = compute_error(A_test,b_test,x.GDstr);
fprintf('Error w.r.t 0-1 loss: %1.3e\n',e.GDstr);
end

parameter.maxit                 = 4000;
if solve.AGD
[x.AGD    , info.AGD       ]    = AGD    (fx, gradf, parameter);
e.AGD = compute_error(A_test,b_test,x.AGD);
fprintf('Error w.r.t 0-1 loss: %1.3e\n',e.AGD);
end

parameter.maxit                 = 2000;
if solve.AGDstr
[x.AGDstr , info.AGDstr    ]    = AGDstr (fx, gradf, parameter);
e.AGDstr = compute_error(A_test,b_test,x.AGDstr);
fprintf('Error w.r.t 0-1 loss: %1.3e\n',e.AGDstr);
end

parameter.maxit                 = 500;
if solve.AGDR
[x.AGDR   , info.AGDR      ]    = AGDR   (fx, gradf, parameter);
e.AGDR = compute_error(A_test,b_test,x.AGDR);
fprintf('Error w.r.t 0-1 loss: %1.3e\n',e.AGDR);
end
     
parameter.maxit                 = 450;
if solve.LSGD
[x.LSGD     , info.LSGD    ]    = LSGD   (fx, gradf, parameter);
e.LSGD = compute_error(A_test,b_test,x.LSGD);
fprintf('Error w.r.t 0-1 loss: %1.3e\n',e.LSGD);
end

parameter.maxit                 = 400;
if solve.LSAGD
[x.LSAGD    , info.LSAGD   ]    = LSAGD  (fx, gradf, parameter);
e.LSAGD = compute_error(A_test,b_test,x.LSAGD);
fprintf('Error w.r.t 0-1 loss: %1.3e\n',e.LSAGD);
end

parameter.maxit                 = 100; 
if solve.LSAGDR
[x.LSAGDR   , info.LSAGDR  ]    = LSAGDR (fx, gradf, parameter);
e.LSAGDR = compute_error(A_test,b_test,x.LSAGDR);
fprintf('Error w.r.t 0-1 loss: %1.3e\n',e.LSAGDR);
end

parameter.maxit                 = 4000;             
if solve.AdaGrad
[x.AdaGrad     , info.AdaGrad        ]    = AdaGrad     (fx, gradf, parameter);
e.AdaGrad = compute_error(A_test,b_test,x.AdaGrad);
fprintf('Error w.r.t 0-1 loss: %1.3e\n',e.AdaGrad);
end

parameter.maxit                 = 4000;             
if solve.ADAM
[x.ADAM     , info.ADAM        ]    = ADAM     (fx, gradf, parameter);
e.ADAM = compute_error(A_test,b_test,x.ADAM);
fprintf('Error w.r.t 0-1 loss: %1.3e\n',e.ADAM);
end


%%%% STOCHASTIC METHODS
[fx, gradf, gradfsto, ~] = Oracles(b, A, sigma);

parameter.no0functions          = n;

parameter.maxit                 = 5*n;
if solve.SGD
[x.SGD       , info.SGD      ]    = SGD     (fx, gradfsto, parameter);
e.SGD = compute_error(A_test,b_test,x.SGD);
fprintf('Error w.r.t 0-1 loss: %1.3e\n',e.SGD);
end

if solve.SAG
[x.SAG       , info.SAG      ]    = SAG     (fx, gradfsto, parameter);
e.SAG = compute_error(A_test,b_test,x.SAG);
fprintf('Error w.r.t 0-1 loss: %1.3e\n',e.SAG);
end

parameter.maxit                 = round(1.5*n);
if solve.SVR
[x.SVR       , info.SVR      ]    = SVR     (fx, gradf, gradfsto, parameter);
e.SVR = compute_error(A_test,b_test,x.SVR);
fprintf('Error w.r.t 0-1 loss: %1.3e\n',e.SVR);
end


fprintf(strcat('Numerical solution process is completed. \n'));
%% plot the results
%close all

%% function values plots of deterministic algos
figure('position', [0 0 1280 550]);
subplot(1, 2, 1);         % wrt # iterations
colors = hsv(13);
legend_lab = {};
if isfield(info,'GD')
loglog(info.GD.fx-fs_opt, 'LineWidth',3,'color',colors(1,:)); hold on;
legend_lab{end+1} = 'GD';
end
if isfield(info,'GDstr')
loglog(info.GDstr.fx-fs_opt, 'LineWidth',3,'color',colors(2,:),'LineStyle',':'); hold on;
legend_lab{end+1} = 'GD-\mu';
end
if isfield(info,'AGD')
loglog(info.AGD.fx-fs_opt, 'LineWidth',3,'color',colors(3,:)); hold on;
legend_lab{end+1} = 'AGD';
end
if isfield(info,'AGDstr')
loglog(info.AGDstr.fx-fs_opt, 'LineWidth',3,'color',colors(4,:),'LineStyle',':'); hold on;
legend_lab{end+1} = 'AGD-\mu';
end
if isfield(info,'LSGD')
loglog(info.LSGD.fx-fs_opt, 'LineWidth',3,'color',colors(5,:)); hold on;
legend_lab{end+1} = 'LSGD';
end
if isfield(info,'LSAGD')
loglog(info.LSAGD.fx-fs_opt, 'LineWidth',3,'color',colors(6,:),'LineStyle',':'); hold on;
legend_lab{end+1} = 'LSAGD';
end
if isfield(info,'AGDR')
loglog(info.AGDR.fx-fs_opt, 'LineWidth',3,'color',colors(7,:)); hold on;
legend_lab{end+1} = 'AGDR';
end
if isfield(info,'LSAGDR')
loglog(info.LSAGDR.fx-fs_opt, 'LineWidth',3,'color',colors(8,:),'LineStyle',':'); hold on;
legend_lab{end+1} = 'LSAGDR';
end
if isfield(info,'AdaGrad')
loglog(info.AdaGrad.fx-fs_opt, 'LineWidth',3,'color',colors(9,:)); hold on;
legend_lab{end+1} = 'AdaGrad';
end
if isfield(info,'ADAM')
loglog(info.ADAM.fx-fs_opt, 'LineWidth',3,'color',colors(12,:)); hold on;
legend_lab{end+1} = 'ADAM';
end

axis tight
ylim([1e-9,1e0])
xlabel('# iteration', 'FontSize',16);
ylabel('f(x) - f^*', 'FontSize',20);
legend(legend_lab, 'FontSize', 15, 'Location', 'best');
grid on


% STOCHASTIC PLOT

subplot(1,2,2);       
colors = hsv(13);
legend_lab = {};
if isfield(info,'GD')
semilogy((0:info.GD.iter-1), info.GD.fx-fs_opt,'-o', 'LineWidth',3,'color',colors(1,:)); hold on;
legend_lab{end+1} = 'GD';
end

if isfield(info,'SGD')
semilogy((0:info.SGD.iter-1)/n, info.SGD.fx-fs_opt, 'LineWidth',3,'color',colors(9,:),'LineStyle',':'); hold on;
legend_lab{end+1} = 'SGD';
end

if isfield(info,'SAG')
semilogy((0:info.SAG.iter-1)/n, info.SAG.fx-fs_opt, 'LineWidth',3,'color',colors(6,:),'LineStyle',':'); hold on;
legend_lab{end+1} = 'SAG';
end

if isfield(info,'SVR')
semilogy((0:info.SVR.iter-1)/n, info.SVR.fx-fs_opt, 'LineWidth',3,'color',colors(3,:),'LineStyle',':'); hold on;
legend_lab{end+1} = 'SVR';
end

axis tight
xlim([0,5])
ylim([1e-4,1e0])
xlabel('#epochs', 'FontSize',16);
ylabel('$f(\mathbf{x}^k) - f^\star$', 'Interpreter', 'latex', 'FontSize',18)
legend(legend_lab, 'FontSize', 15, 'Location', 'best');
grid on
