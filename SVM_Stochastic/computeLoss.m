function loss = computeLoss(X,Y,w,lambda)
[~,m] = size(X);
loss = lambda/2*norm(w(2:end))^2+1/m*sum(max(0,1-Y.*(w'*X)));
end

