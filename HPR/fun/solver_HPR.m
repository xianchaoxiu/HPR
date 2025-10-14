function [Relerrs,Rx,x,i] = solver_HPR( Amatrix,y,x_orig,alphaa,lambda_cv)
 %% Inputs:

%% Outputs:

[n, p]=size(Amatrix);
Amatrixt=Amatrix.';
f00=sum(max(min(y.^2/2, alphaa^2/2),alphaa*abs(y)-alphaa^2/2))/n;

npower_iter = 50;                           % Number of power iterations
z0 = randn(p,1); z0 = z0/norm(z0,'fro');    % Initial guess
for tt = 1:npower_iter,                     % Power iterations
    z0 = Amatrixt*(y.* (Amatrix*z0)); z0 = z0/norm(z0,'fro');
end
normest = sqrt(sum(y)/numel(y));    % Estimate norm to scale eigenvector
x = normest * z0;

maxiter=1000; 
Relerrs=zeros(maxiter+1,1);
Relerrs(1)=norm(x_orig - exp(-1i*angle(trace(x_orig'*x))) * x, 'fro')/norm(x_orig,'fro');

eps=1e-8;epsx=1e-6;

%%
tic;  
Rf=[];Rx=[];OBJ    = zeros(5,1); 
for i=1:maxiter
  x0=x;
          Gx = gradienth(Amatrix,Amatrixt,alphaa,y,x0);   ngx=norm(Gx);
        f0 = gun(Amatrix,alphaa,lambda_cv,y,x0);               
            mu = .9; 
    for j = 1 : 20
        x_tem=x0-mu*Gx;
         x=Hvp(x_tem,2*lambda_cv*mu); 
         f = gun(Amatrix,alphaa,lambda_cv,y,x);nxx02=norm(x-x0)^2;
        if f  <= f0 - mu*nxx02/1e5; break; end
        mu = mu/5;
    end 
    

%         Rf=[Rf,f];
        Relerrs(i+1)=norm(x_orig - exp(-1i*angle(trace(x_orig'*x))) * x, 'fro')/norm(x_orig,'fro');Relerrs(i+1);
        
        Rx=[Rx,x];
        criteria=norm(x0-x)/max(1,norm(x0));
        %fprintf('iter1: %2d; criteria: %2.3e; \n',i, criteria)
        currenterrs=Relerrs(i+1);
           if ngx<epsx || criteria<eps 
            break; end 
        
  
end
for i=i:maxiter
    Relerrs(i+1)= currenterrs;
end
time=toc;
end


%%
function [g] = gun(Amatrix,alphaa,lambda_cv,y,x)
n=size(Amatrix,1);
Ax=Amatrix*x;
yAx2=(abs(Ax)).^2-y;xby2=(yAx2).^2/2;xby1a=alphaa*abs(yAx2)-alphaa^2/2;
fvec=max(min(xby2, alphaa^2/2),xby1a);
g=sum(fvec)/n+lambda_cv*norm(x,.5)^(.5);
end


function [ grad ] = gradienth( Amatrix,Amatrixt,alphaa,y,x )
m=size(Amatrix,1);
p=size(Amatrix,2);
Ax=Amatrix*x; yAx=abs(Ax).^2-y;
c=Ax.*min(alphaa,max(yAx,-alphaa));
b=repmat(c.',p,1);
B=b.*conj(Amatrixt);
grad=2*(sum(B,2))/m;

end



function [ h ] =Hvp( x,v)
%% Half thresholding operator
p=length(x); h=zeros(p,1);
for i=1:p
xj=x(i);
    if abs(xj)<=(54^(1/3)/4)*(v.^(2/3))
        h(i)=0;
    else
        h(i)=(2/3)*xj*(1+cos((2/3)*pi-(2/3)*acos((v/8)*(abs(xj)/3)^(-3/2))));
    end
end
end