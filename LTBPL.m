function [result]=LTBPL(X,alpha,beta,omega,gt)
C = size(unique(gt),1);
V = size(X,2); %number of views
N = size(X{1},2);% number of data points
NITER = 20;
%normalized X
for i=1:V
    X{i} = X{i}./repmat(sqrt(sum(X{i}.^2,1)),size(X{i},1),1);
end
%Initilize A,tensor J,multiplier Q
for i = 1:V
    A{i} = constructW_PKN(X{i},10);
    ed{i} = L2_distance_1(X{i}, X{i});
    J{i} = zeros(N,N);
    Q{i} = zeros(N,N);  %multiplier
end

w = 1/V*ones(1,V); 
sX = [N, N, V];  
pho = 0.1; 
mu = 2;
lambda = 0.1;

%% outer loop
for iter = 1:NITER
    % == update S ==
    if iter ==1
       [Y, S] = CLR(w,A,C,lambda);
    else
       [Y, S] = CLR(w,A,C,lambda,S0);
    end
    S0 = S;  %after iter=1,we have S0
    
    % == update A{i} ==
    for i = 1:V
        temp_A=zeros(N);
        B = J{i}-Q{i}/pho;
        tmp_ed=ed{i};
        tmp_w=w(1,i);
        for j = 1:N
            ad = (2*tmp_w*S(j,:)+pho*B(j,:)-tmp_ed(j,:))/(2*alpha+2*tmp_w+pho);
            temp_A(j,:) = EProjSimplex_new(ad);
        end
        A{i} = temp_A;
    end
    
    % == update J{i} ==
    A_tensor = cat(3, A{:,:});
    Q_tensor = cat(3, Q{:,:});
    a = A_tensor(:);
    q = Q_tensor(:);
    [j, ~] = wshrinkObj(a+1/pho*q,beta/pho,sX,0,3,omega);
    J_tensor = reshape(j, sX); 
    for i=1:V
        J{i} = J_tensor(:,:,i);
    end
    
    % update alpha
    for i = 1:V
        w(1,i) = 0.5/norm(S-A{i},'fro');
    end
    
   % == update Q{i} ==
    for i=1:V
        Q{i} = Q{i}+pho*(A{i}-J{i});
    end
    
    % == update pho ==
    pho = pho*mu;
      
end

[ACC,NMI,PUR] = ClusteringMeasure(gt,Y); %ACC NMI Purity
[Fscore,Precision,R] = compute_f(gt,Y);
[AR,~,~,~]=RandIndex(gt,Y);
result = [ACC NMI PUR Fscore Precision R AR];

