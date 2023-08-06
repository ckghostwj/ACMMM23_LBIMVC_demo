% If you use the code, please cite the following paper:
% Jie Wen, et al. 
% Localized and Balanced Efficient Incomplete Multi-view Clustering[C]. 
% Proceedings of the 31st ACM International Conference on Multimedia, 2023.
% For any questions, contact me via jiewen_pr@126.com

clear all
clc

Dataname = '3sources3vbigRnSp';

percentDel = 0.5;
lambda = 0.1; 
miu = 0.01;
rho = 1.02;
para_r = 3;
para_k = 15;

Datafold = [Dataname,'_percentDel_',num2str(percentDel),'.mat'];
load(Dataname); 
load(Datafold);
[numFold,numInst] = size(folds);

numClust = length(unique(truth));
numInst  = length(truth); 


f = 3;
load(Dataname);
load(Datafold);
ind_folds = folds{f};                   
truthF = truth;

if size(X{1},2)~=length(truth) || size(X{2},2)~=length(truth)
    for iv = 1:length(X)
        X{iv} = X{iv}';  % 一列一个样本
    end
end
clear truth

clear Y
for iv = 1:length(X)
    X1 = X{iv};
    X1 = NormalizeFea(X1,0);
    ind_1 = find(ind_folds(:,iv) == 1);
    ind_0 = find(ind_folds(:,iv) == 0);
    X1(:,ind_0) = []; 
    Y{iv} = X1;

    linshi_G = diag(ind_folds(:,iv));
    linshi_G(:,ind_0) = [];
    G{iv} = linshi_G;

    options = [];
    options.NeighborMode = 'KNN';
    options.k = para_k;
    options.WeightMode = 'HeatKernel';      % Binary  HeatKernel
    linshi_W = full(constructW(X1',options))+eye(size(X1,2));
    W_graph{iv} = (linshi_W+linshi_W')*0.5;
end
clear X
X = Y;
clear Y;

max_iter = 100;
[Con_P,obj] = LBIMVC(X,W_graph,G,ind_folds,numClust,lambda,para_r,miu,rho,max_iter);

[~,pre_labels] = max(Con_P,[],1);
result_cluster = ClusteringMeasure(truthF, pre_labels)*100

