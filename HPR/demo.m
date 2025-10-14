%% test for Median-TWF, presented in paper "Provable Nonconvex Phase Retrieval with Outliers: Median Truncated Wirtinger Flow" @Huishuai Zhang, Yuejie Chi and Yingbin Liang
% The code is adapted from implementation of TWF by Y. Chen and E. J. Cand鑣 and Wirtinger flow  by E. Candes, X. Li, and M. Soltanolkotabi
clear;clc;
clc;clear all;
%% Set Parameters
if exist('Params')                == 0,  Params.n2          = 1;    end
if isfield(Params, 'n1')          == 0,  Params.n1          = 128;  end             % signal dimension
%if isfield(Params, 'm')           == 0,  Params.m           = 3* Params.n1;  end     % 2:6倍n1 改
if isfield(Params, 'cplx_flag')   == 0,  Params.cplx_flag   = 1;     end             % real: cplx_flag = 0;  complex: cplx_flag = 1;改
if isfield(Params, 'grad_type')   == 0,  Params.grad_type   = 'Outliers';  end     % 'TWF_Poiss': Poisson likelihood
%Huber参数
if isfield(Params, 'alphaa')    == 0,  Params.alphaa    =1.345;  end
if isfield(Params, 'gammaa')    == 0,  Params.gammaa    = 0.5;    end

if isfield(Params, 'alpha_lb')    == 0,  Params.alpha_lb    = 0.3;  end
if isfield(Params, 'alpha_ub')    == 0,  Params.alpha_ub    = 5;    end
if isfield(Params, 'alpha_h')     == 0,  Params.alpha_h     = 12;   end
if isfield(Params, 'alpha_y')     == 0,  Params.alpha_y     = 3;    end 
if isfield(Params, 'T')           == 0,  Params.T           = 300;  end     % Params.T for median-TWF  %number of iterations WF: Params.T=2500; 其他：Params.T=500；
if isfield(Params, 'mu')          == 0,  Params.mu          = 0.2;  end		% step size / learning parameter
if isfield(Params, 'npower_iter') == 0,  Params.npower_iter = 50;   end		% number of power iterations

%   p           = Params.n1;    
%   n           = Params.m ;     
cplx_flag	= Params.cplx_flag;  % real-valued: cplx_flag = 0;  complex-valued: cplx_flag = 1;    
display(Params) 

%% Make signal and data (noiseless)
%% Make signal and data (noiseless)
Rate=[];%成功率
At1=[];%平均时间
Relerrs6=[];
t11=[];
s=12;
t=[1 2 3 4 5 6 7 8 9 10];
for i=1:length(t)
    n=t(i)*128;ttt=i;
fprintf('n: %4d; \n',n)
p=Params.n1;
rate=0;SucRate=[];Res_huber=[];
t1=[];
noS=5;
 for S    = 1:noS  
 p=Params.n1;
 m=n;
I = randperm(p); SU = I(1:s);
    B =randn(s,1)+cplx_flag*1i*randn(s,1);
    x_orig=zeros(p,1) + 1i*zeros(p,1);
    x_orig(SU) =B;
Amatrix = (randn(n,p) + cplx_flag * 1i * randn(n,p)) / (sqrt(2)^cplx_flag);
 A  = @(I) Amatrix  * I;
At = @(Y) Amatrix' * Y;
y  = abs(A(x_orig)).^2; %------------------------------------------------------------------------变化
ita=0.01;%0.01;0.001;改 
NoiseNorm=ita*norm(y)/sqrt(numel(y));
AWLN = rand(size(y),class(y));
neg = AWLN < 0.5;
AWLN(neg) = (NoiseNorm/sqrt(2)).*log(2.*AWLN(neg));
AWLN(~neg) = (-NoiseNorm/sqrt(2)).*log(2.*(1-AWLN(~neg)));
y=y+AWLN;
%% Check results and Report Success/Failure
alphaa=1.345;%小
tic;
lambda_cv=norm(Amatrix,2)/5000;

[Relerrs,Rx,x,i] = solver_HPR( Amatrix,y,x_orig,alphaa,lambda_cv);

time1=toc;time1,
Relerrs_huber=norm(x_orig - exp(-1i*angle(trace(x_orig'*x))) * x, 'fro')/norm(x_orig,'fro');Relerrs_huber
tic;[ttt S]


Res_huber=[Res_huber;Relerrs_huber];

 t1=[t1;time1];
 
end
 t11=[t11 t1];
 
 Relerrs6=[Relerrs6 Res_huber];
 
end

iter=noS;
At1=[sum(t11(:,1));sum(t11(:,2));sum(t11(:,3));sum(t11(:,4));sum(t11(:,5));sum(t11(:,6));sum(t11(:,7));sum(t11(:,8));sum(t11(:,9));sum(t11(:,10))]/iter;

Rate=[length(find(Relerrs6(:,1)<5*1e-3))/iter;length(find(Relerrs6(:,2)<5*1e-3))/iter;length(find(Relerrs6(:,3)<5*1e-3))/iter;length(find(Relerrs6(:,4)<5*1e-3))/iter;length(find(Relerrs6(:,5)<5*1e-3))/iter;length(find(Relerrs6(:,6)<5*1e-3))/iter;length(find(Relerrs6(:,7)<5*1e-3))/iter;length(find(Relerrs6(:,8)<5*1e-3))/iter;length(find(Relerrs6(:,9)<5*1e-3))/iter;length(find(Relerrs6(:,10)<5*1e-3))/iter];

Time_com=At1'
Rel_com=mean(Relerrs6,1)



figure,
semilogy(1:1:10,Rate,'-*r','LineWidth',2.5), hold on;  
set(gca,'XTick',[1:1:10]);
set(gca,'YTick',[0:0.1:1]);
xlabel('n/p'), ylabel('Success rate'), ...
title('n/p vs. Success rate')
%legend('This work')
hold off;
