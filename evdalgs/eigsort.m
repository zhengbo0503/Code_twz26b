function [Vs,Ds] = eigsort(A)
[V,D] = eig(A);
[~,ind] = sort(diag(D),'descend');
Ds = D(ind,ind);
Vs = V(:,ind);
end
