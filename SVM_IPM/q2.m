%========================================================================%
%                 Welcome, Professor Scheinberg!                         %
%========================================================================%
%There are 4 arguments in function q2: C, ktype, kparam, ns:
% C: Box constraint constant.
% ktype: Kernel type: 'l':linear;'p':polynomial;'g':gaussian;
% kparam: Kernal parameters: degree for polynomial kernel, or dominator for
%         gaussian kernel;
% ns: Normalize the feature when ns == 1.
% Example: to train the SVM for the optical character recognition using 
%          original data with 2nd degree polynomial kernel and box 
%          constraint=5, just type:
% >> hw2(5,'p',2,false)
%========================================================================%

function result = q2(C,ktype,kparam,ns)
traindata = csvread('tra.csv');
global X;
global Y;
siz = size(traindata);
X = traindata(:,1:siz(2)-1)';
Y = traindata(:,siz(2));
[R,IA,IC] = unique(X','rows');
X = R';
if ns == 1
    X = (X-8)/16;
end
Y = Y(IA,:);
Y = (Y==5);
Y=Y*2-1;
sol = svm(X,Y,C,ktype,kparam);
alpha = sol{1};
b = sol{2};
supvec = sol{3};
ay = sol{4};
testdata = csvread('tes.csv');
sizt = size(testdata);
Xt = testdata(:, 1:sizt(2)-1)';
if ns == 1
    Xt = (Xt-8)/16;
end
Yt = testdata(:,sizt(2));
Yt = (Yt==5);
Yt=Yt*2-1;
func_pred = @(x) ay'*kernel(X,x,ktype,kparam)+b;
pret = [];
for i = 1:sizt(1)
    pret = [pret;func_pred(Xt(:,i))];
end
epsilon = (-Yt).*pret+1;
hingeloss = max(0,epsilon);
hingelosssum = sum(hingeloss)
binloss = (sign(pret)~=Yt);
binlosssum = sum(binloss)
accuracy = (1-binlosssum/sizt(1))*100
result = [hingelosssum binlosssum accuracy];
end