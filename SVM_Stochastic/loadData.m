traindata = csvread('tra.csv');
testdata = csvread('tes.csv');
ns = 0;
[m,n] = size(traindata); n = n-1;
[mt,nt] = size(testdata); nt = nt-1;
X = traindata(:,1:n)'; Y = traindata(:,n+1);
Xt = testdata(:,1:nt)'; Yt = testdata(:,nt+1);
if ns ==1 
    X = X/16;
    Xt = Xt/16;
end
[R,IA,IC] = unique(X','rows');
X = [ones(1,m);R']; n = n+1; Y = Y(IA,:);Y = (Y==5)';Y=Y*2-1;
Xt = [ones(1,mt);Xt]; nt = nt+1; Yt = (Yt ==5)'; Yt = Yt*2-1;
w = zeros(n,1);
