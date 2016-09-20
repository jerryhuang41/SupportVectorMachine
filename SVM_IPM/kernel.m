function k = kernel(a,b,type,param)
    sa = size(a);
    sb = size(b);
    k = zeros(sa(2),sb(2));
    if type == 'l'
        k = a'*b;
    elseif type == 'p'
        k = (1+ a'*b).^param;
    elseif type == 'g'
        for i = 1:sa(2)
            for j = 1:sb(2)
                k(i,j) = exp(-norm(a(:,i)-b(:,j))^2/(40*param));
            end
        end
%         if isvector(b)
%             d = sum(gsubtract(a,b).^2)';
%             k = exp(-d/param);
%         elseif a == b
%             k = exp(-squareform(pdist(a'))/param);
%         end
    else
        k = a'*b;
        disp('Error: kernel argument does not exist. Use Linear Kernel as default.')
    end
end

