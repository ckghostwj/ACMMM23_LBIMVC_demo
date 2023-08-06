function [Con_P,obj] = LBIMVC(X,W_graph,G,ind_folds,numClust,lambda,para_r,miu,rho,max_iter)
% If you use the code, please cite the following paper:
% Jie Wen, et al. 
% Localized and Balanced Efficient Incomplete Multi-view Clustering[C]. 
% Proceedings of the 31st ACM International Conference on Multimedia, 2023.
% For any questions, contact me via jiewen_pr@126.com

Nsamp = size(ind_folds,1);

alpha = ones(length(X),1);
alpha_r = alpha.^para_r;

seed_num = 3737;
rand('twister',seed_num)
% rand('seed',seed_num)
Con_P = abs(rand(numClust,Nsamp));
Con_P = Con_P./repmat(sum(Con_P,1),numClust,1);
Con_Q = Con_P;

NormX = 0;
for iv = 1:length(X)
    rand('twister',seed_num+iv*20);
%     rand('seed',seed_num+iv*20);
    linshi_U = rand(size(X{iv},1),numClust);
    U{iv} = orth(linshi_U);
    NormX = NormX+norm(X{iv},'fro')^2;
end

Con_C = 0;
for iter = 1:max_iter
    
    % ---------------- Con_P ---------- %
    linshi_PWG = zeros(numClust,Nsamp);
    linshi_GDG2 = zeros(Nsamp,1);
    for iv = 1:length(X)
        graph_A = sum(W_graph{iv},2);
        aa = alpha_r(iv)*graph_A;
        ind_1 = find(ind_folds(:,iv)==1);
        cc = zeros(Nsamp,1);
        cc(ind_1) = aa;
        linshi_GDG2 = linshi_GDG2+cc;
        linshi_PWG = linshi_PWG+alpha_r(iv)*(U{iv}'*X{iv}*W_graph{iv}'*G{iv}');
    end 
    linshi_Con_P = (lambda*linshi_PWG+0.5*(miu*Con_Q-Con_C))*diag(1./max(lambda*linshi_GDG2+0.5*miu,0));
    for in = 1:size(Con_P,2)
        temp1 = linshi_Con_P(:,in);
        Con_P(:,in) = EProjSimplex_new(temp1);
    end    
       
    % -------- U ------ %
    for iv = 1:length(X)
        temp = X{iv}*W_graph{iv}'*G{iv}'*Con_P';
        temp(isnan(temp)) = 0;
        temp(isinf(temp)) = 0;        
        [linshi_U,~,linshi_V] = svd(temp,'econ');      
        linshi_U(isnan(linshi_U)) = 0;
        linshi_U(isinf(linshi_U)) = 0;    
        linshi_V(isnan(linshi_V)) = 0;
        linshi_V(isinf(linshi_V)) = 0; 
        U{iv} = linshi_U*linshi_V';  
    end
    % ------- Q --------%
    Con_Q = 1/(miu^2+Nsamp*miu)*((miu*Con_P+Con_C)*((miu+Nsamp)*eye(Nsamp,Nsamp)-ones(Nsamp,Nsamp)));
    Con_Q = max(Con_Q,0);
    
    % ----------- alpha--------%
    for iv = 1:length(X)
        graph_D = diag(sum(W_graph{iv}));
        Rec_error(iv) = trace(X{iv}*graph_D*X{iv}')+trace(Con_P*G{iv}*graph_D*G{iv}'*Con_P')-2*trace(U{iv}*Con_P*G{iv}*W_graph{iv}*X{iv}');
    end
    aH = bsxfun(@power,Rec_error, 1/(1-para_r));     % h = h.^(1/(1-r));
    alpha = bsxfun(@rdivide,aH,sum(aH)); % alpha = H./sum(H);
    alpha_r = alpha.^para_r;   
    
    % -------- C miu -----%    
    leq3 = Con_P-Con_Q;
    stopC = max(max(abs(leq3)));
    if stopC>1e-6
        Con_C = Con_C+miu*(Con_P-Con_Q);    
        miu = min(rho*miu,1e8); 
    else
        break
    end 
    
    % ------- obj ------- %
    obj(iter) = (alpha_r*Rec_error'*lambda+0.5*trace(Con_P*ones(Nsamp,Nsamp)*Con_P'))/NormX;
    if iter > 3 && abs(obj(iter)-obj(iter-1))<1e-4
        iter
        break;
    end
 
end