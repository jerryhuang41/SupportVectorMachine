function sol = svm(X,Y,C,ktype,kparam)
siz = size(X);
n = siz(1);
m = siz(2)
K = kernel(X,X,ktype,kparam);
q = (Y*Y').*K;
% disp('Q computation completed')
% ue = max(eig(q));
% le = min(eig(q));
% connum = max(eig(q))/min(eig(q));
q = q + (0.00001/C)*eye(m);
c = -ones(m,1);
a = Y';
blc = 0;
buc = 0;
blx = 0*ones(m,1);
bux = C*ones(m,1);
tic

[res] = mskqpopt(q,c,a,blc,buc,blx,bux);
toc
sol = res.sol;
alpha = sol.itr.xx;

ay = alpha.*Y;
supvec = find((alpha>10^(-4))&(alpha<C-10^(-4)));
b = sol.itr.suc-sol.itr.slc;
sol = {alpha,b,supvec,ay};


%===plotting things for 2 dimension only===%
griddot = 100;
if n == 2
    stp1 = (0.1+range(X(1,:)))/griddot;
    stp2 = (0.1+range(X(2,:)))/griddot;
    x1 = min(min(X))-0.1*griddot*stp1:stp1:max(max(X))+0.1*griddot*stp1;
    x2 = min(min(X))-0.1*griddot*stp2:stp2:max(max(X))+0.1*griddot*stp2;
    [x1,x2] = meshgrid(x1,x2);
    predict = @(x_,y_) ay'*kernel(X,[x_;y_],ktype,kparam)+b;
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
    aa = 16+2.4*(range(X(1,:))*range(X(2,:)))/m;
    %===============================
    scatter(xp1,xp2,aa,[0.4 0.3 0.6],'filled')
    scatter(xn1,xn2,aa,[1 0.6 0.2],'filled')
    scatter(xv1,xv2,0.6*aa,[1 1 1],'filled','^')
end
end
