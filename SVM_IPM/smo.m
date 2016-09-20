% X =[4    3    4  3 6 -2;
%     1    2    2  4 5 1];
% Y = [1;  -1;  1;-1;1;-1];
X = [1  0   0.5  1.5; 
     1  2   0.25 1.5; ];
Y = [1; -1; -1;  1];
traindata = csvread('tra.csv');
global X;
global Y;

%X = traindata(:,1:siz(2)-1)';
%Y = traindata(:,siz(2));
[R,IA,IC] = unique(X','rows');
X = R';
Y = Y(IA,:);
%Y = (Y==5);
%Y = Y*2-1;
siz = size(X);
C = 5;
ktype = 'l';
kparam = 2;
n = siz(1);
m = siz(2);
itemax = 2;
K = kernel(X,X,ktype,kparam);
%q = (Y*Y').*K;
alpha = zeros(m,1);
wolfe = [];
e = ones(m,1);
b = 0;
%func_pred = @(x) ay'*kernel(X,x,ktype,kparam)+b;
i = 1;
j = 2;
tol = 10^(-3);
vio = ones(m,1);
ay = alpha.*Y;
wolfe = sum(alpha)-0.5*(ay'*(X'*X)*ay)
for k = 1:itemax
    for p=1:m
        ay = alpha.*Y;
        fx = (ay'*K+b)';
        e = fx-Y;
%         bi = e(i)+Y(i)*(alpha(i)-alphaiold)*K(i,i)+Y(j)*(alpha(j)-alphajold)*K(i,j)+b;
%         bj = e(j)+Y(i)*(alpha(i)-alphaiold)*K(i,j)+Y(j)*(alpha(j)-alphajold)*K(j,j)+b;
%         b = (bi+bj)/2;
        if alpha(i)<C&&alpha(i)>0
            b = Y(i)-ay'*kernel(X,X(:,i),ktype,kparam);
        end
        yf = Y.*fx;
        tempa = alpha(p);
        tempb = yf(p);
        alpha(alpha<tol)=0;
        alpha(alpha>C-tol) = C;
        if (tempa==0&&tempb>=1)||(tempa>0&&tempa<C&&tempb==1)||(tempa ==C&&tempb<=1)
            vio(p) = 0;
        else
            vio(p) = 1;
            j = p;
            if e(j) > 0
                i = find(e == min(e));
                i = i(1);
            else
                i = find(e == max(e));
                i = i(1);
            end
            if i == j
                i = mod(j-1+randi(m-1,1),m)+1;
            end
            wolfe = cat(2,wolfe,e'*alpha-0.5*ay'*K*ay);
            zeta = ay(i)+ay(j);
            temayy = -ay*Y(j);
            temayy([i j]) = 0;
            Z = K(:,i)-K(:,j);
            omega = 1-Y(i)*Y(j)+zeta*Y(j)*(K(i,i)-K(i,j))-temayy'*Z;
            upd = omega/(K(i,i)-2*K(i,j)+K(j,j));
            if Y(i)*Y(j) == 1
                H = min(zeta*Y(i),C);
                L = max(zeta*Y(i)-C,0);
            else
                H = min(C-Y(i)*zeta,C);
                L = max(-zeta*Y(i),0);
            end
            alphajold = alpha(j);
            if upd > H
                alpha(j) = H;
            elseif upd <L
                alpha(j) = L;
            else
                alpha(j) = upd;
            end
            alphaiold = alpha(i);
            alpha(i) = Y(i)*(zeta-alpha(j)*Y(j));
            alpha(alpha<tol)=0;
            alpha(alpha>C-tol) = C;
        end
    wolfe = sum(alpha)-0.5*(ay'*(X'*X)*ay)
    end
    if max(vio) ==0
        disp('yay!');
        disp(max(vio));
        disp(k);
        ay = alpha.*Y;
        if alpha(i)<C&&alpha(i)>0
            b = Y(i)-ay'*kernel(X,X(:,i),ktype,kparam);
        end
    end
end
        
%         if abs(e(i)-e(j))<10^(-8)
%             disp('small advance');
%             i =mod(k-1,m)+1;
%             j = mod(k-1+randi(m-1,1),m)+1;
%         end
%================================
supvec = find((alpha>0)&(alpha<C));
%b = Y(supvec(1))-ay'*kernel(X,X(:,supvec(1)),ktype,kparam);
func_pred = @(x) ay'*kernel(X,x,ktype,kparam)+b;
%==================
    griddot = 100;
if n == 2
    step1 = (0.1+range(X(1,:)))/griddot;
    step2 = (0.1+range(X(2,:)))/griddot;
    x1 = min(min(X))-0.1*griddot*step1:step1:max(max(X))+0.1*griddot*step1;
    x2 = min(min(X))-0.1*griddot*step2:step2:max(max(X))+0.1*griddot*step2;
    [x1,x2] = meshgrid(x1,x2);
    predict = @(x,y) ay'*kernel(X,[x;y],ktype,kparam)+b;
    z = arrayfun(predict,x1,x2);
    figure(); hold on;
    contour(x1,x2,z,[0 0],'k-');
    contour(x1,x2,z,[-1 1],'k--');
    xpt1 = X(1,:)';
    xp1 = xpt1(Y==1);
    xpt2 = X(2,:)';
    xp2 = xpt2(Y==1);
    xnt1 = X(1,:)';
    xn1 = xnt1(Y==-1);
    xnt2 = X(2,:)';
    xn2 = xnt2(Y==-1);
    %=========support vectors============
    xv1 = X(1,:)';
    xv1 = xv1(supvec);
    xv2 = X(2,:)';
    xv2 = xv2(supvec);
    xr = range(X(1,:));
    yr = range(X(2,:));
    aa = 40+2.4*(range(X(1,:))*range(X(2,:)))/m;
    %===============================
    scatter(xp1,xp2,aa,[0.4 0.3 0.6],'filled')
    scatter(xn1,xn2,aa,[1 0.6 0.2],'filled')
    scatter(xv1,xv2,0.6*aa,[1 1 1],'filled','^')
end
%==============on test set=============
% testdata = csvread('tes.csv');
% sizt = size(testdata);
% Xt = testdata(:, 1:sizt(2)-1)';
% Yt = testdata(:,sizt(2));
% Yt = (Yt==5);
% Yt=Yt*2-1;
% pret = [];
% for i = 1:sizt(1)
%     pret = [pret;func_pred(Xt(:,i))];
% end
% epsilon = (-Yt).*pret+1;
% hingeloss = max(0,epsilon);
% hingelosssum = sum(hingeloss)
% binloss = (sign(pret)~=Yt);
% binlosssum = sum(binloss)
% accuracy = (1-binlosssum/sizt(1))*100