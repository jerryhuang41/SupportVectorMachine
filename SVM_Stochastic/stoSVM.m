%use stochastic update to train SVM

function [w,lossRec] = stoSVM(X,Y,C,a,p,itemax)
if nargin <6
    itemax = 1000;
    if nargin <5
        p = 30;
        if nargin<4
            a = 1;
        end
    end
end
[n,m] = size(X);
avgw = zeros(n,1);
w = avgw;
lambda = 1/(C*m);
lossRec = [];
tailCut = 200;
ii = 0;
for k = 1:itemax
    pac = dividerand([Y;X],p/m,0,1-p/m);
    pX = pac(2:end,:); pY = pac(1,:);
    u = a/(lambda*k);
    wtmp = w-u*gradient(pX,pY,w,lambda);
    w = wtmp*min(1,1/sqrt(lambda)/norm(wtmp));
    lossRec = cat(2,lossRec,computeLoss(X,Y,w,lambda));
%     if k>itemax-tailCut
%         avgw = avgw+1/tailCut*w;
%     end
    avgw = avgw+1/itemax*w;
    
end
end

