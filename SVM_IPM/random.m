%========================================================================%
%                 Welcome, Professor Scheinberg!                         %
%========================================================================%

%===============Adjust the parameters here===============================%
%Run this script to see how SVM performs using randomly generated data.
%!!The random data is multi-normally distributed.
n = 2; %Dimension of features. Plotting only available for n==2;
m = 30; %Numbers of data points generated.
C = 5; %Box constraint constant.
ktype = 'p'; %Kernel type: 'l':linear;'p':polynomial;'g':gaussian;
kparam = 2; %Kernal parameters: degree for polynomial kernel, or dominator for gaussian kernel;

sigmod = 3; %Coefficient of the covariance matrix of the random data point. 
%           sigma<0.5 almost guaranteed linear seperatable data.

regen = 1; %If regen == 1, it generates random data every time you run the script.
%           If you like the current data, set it to 0 to keep the data
%           unchanged.


%====================Don't worry about anything under this line==========%
if regen == 1
    A = generate(n,m,sigmod);
elseif regen == 0
end
global X
global Y
X = A(1:n,:);
Y = A(n+1, :)';
[R,IA,IC] = unique(X','rows');
X = R';
Y = Y(IA,:);
sol = svm(X,Y,C,ktype,kparam);
alpha = sol{1};
b = sol{2};
supvec = sol{3};
disp('The support vectors are:');
disp(supvec')