
function[A] = generate(n,m,sigmod)
a = rand(n)-0.5*ones(n);
X = rand(n,m);
Y = ones(1,m);
mup = (rand(n,1)-0.4)*rand(1,1)*30;
mun = (rand(n,1)-0.6)*rand(1,1)*30;
sigma = sigmod/5*(a.'*a*abs((rand(1,1)-0.5)*2) + rand(1,1)*eye(n)*3)*norm(mup-mun);
    for i = 1:m
        if rand(1,1)>0.5
            Y(i) = 1;
            X(:,i) = mvnrnd(mup,sigma).';
        else
            Y(i) = -1;
            X(:,i) = mvnrnd(mun,sigma).';
        end;
    end;
  A = cat(1,X,Y);
end