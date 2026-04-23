function normf = off(A,bypass)
%OFF(A) Frobenius norm of matrix's off diagonal entries
%   normf = OFF(A) provides a measure on the size of the off diagonal of a
%   symmetric matrix. 
%
%   See also NORM

% Check input
if nargin < 2, bypass = true; end
if ~issymmetric(A) && ~bypass, error("Input should be symmetric"); end

% Parameters 
n = size(A,1);

% Main 
A(1:n+1:n*n) = 0;
normf = norm(A,'fro');
end
