function result = gradient(X,Y,w,lambda)
[~,m] = size(X);
indice = 1-Y.*(w'*X);
Y(indice<=0)=0;
result = lambda*[0;w(2:end)]-1/m*(X*Y');
%lambda*[0;w(2:end)]+
end