function [U,S,V,nos,scalecond] = mposj_ssd(G)
%MPOSJ_SSD - Mixed precision one-sided Jacobi algorithm that uses single and
%        double precision
%
%   Usage:
%       [U,S,V] = MPOSJ_SSD(G)
%       [U,S,V,nos] = MPOSJ_SSD(G)
%       [U,S,V,nos,scalecond] = MPOSJ_SSD(G)
%
%   Purpose: 
%       MPOSJ_SSD computes a SVD of a general matrix based on LAPACK routine
%       DGESVJ, the one-sided Jacobi SVD algorithm. 
%       MPOSJ_SSD first computes a preconditioner at single precision,
%		applies the preconditioner at double precision, and finally performs
%		DGESVJ on the preconditioned matrix at double precision. 
%       For ill-conditioned matrix, MPOSJ_SSD can compute singular values 
%		with smaller relative forward error compared to MATLAB function svd.
%
%   Arguments: 
%       (1) G - Real, double matrix.
%   
%   Outputs:
%       (1) U,S,V - Real, double matrix.
%           Suppose G is mxn, then U is mxn whose columns are numerically
%           orthonormal, S is a nxn diagonal matrix whose diagonal entries
%           are singular values of G, and V is a nxn numerically orthogonal
%           matrix. 
%       (2) nos - Integer.
%           Number of sweep used by DGESVJ.
%       (3) scalecond - Real, double.
%           Scaled condition number of the preconditioned matrix. Useful
%           for some posterior analysis. 
%
%   Author:
%       Zhengbo Zhou, Manchester, UK, Dec 2025
%

[m,n] = size(G);
if m < n   
    [Vt,St,Ut,nos,scalecond] = mposj_ssd(G');
    U = Ut; S = St; V = Vt;
    return
end

idty = eye(n);
doqr = (m >= (11*n)/6); % Consistent with LAPACK choice. 

% Compute the preconditioner at single precision
[~,~,Vt] = svd(single(G),'econ');

% Precondition the matrix G at double precision
Gt = single(double(G)*double(Vt)); 

% Output scaled condition number for posterior analysis. 
if nargout >= 5
    scalecond = scond(Gt); 
end

% Apply the one-sided Jacobi
if doqr % Apply QR factorization before doing Jacobi SVD
    [Qgt,Rt] = qr(Gt,'econ');
    optlwork = max(6,m+n);
    work = zeros(optlwork,1,'single');
    [U,S,V,~,work,info] = sgesvj_mex(Rt,'U','U','V',n,idty,optlwork,work);
    nos = work(4);
    U = Qgt*U;
else % Plain Jacobi SVD
    optlwork = max(6,m+n);
    work = zeros(optlwork,1,'single');
    [U,S,V,~,work,info] = sgesvj_mex(Gt,'G','U','V',n,idty,optlwork,work);
    nos = work(4);
end

if info < 0
    error("SGESVJ has invalid inputs.\n");
elseif info > 0
    warning("SGESVJ does not converged in 30 iterations.\n")
end

V = Vt*V;
end
