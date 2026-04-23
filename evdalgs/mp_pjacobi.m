function [V,D,NROT,NSWEEP,BOUND,SCOND] = mp_pjacobi(A, method)
%MP_PJACOBI - Mixed-precision preconditioned Jacobi algorithm
%	
%   Usage:
%      [V,D,NROT,NSWEEP,BOUND,SCOND] = MP_PJACOBI(A)
%   
%   Purpose:
%       MP_PJACOBI computes the spectral decomposition (SPD) of a real, 
%       N-by-N matrix A using the mixed-precision preconditioned Jacobi
%       algorithm. The SPD of A is written as 
%               A = V * diag(D) * V'
%       where V is an N-by-N orthogonal matrix, and D is an N dimensional
%       array whose entries are eigenvalues in descending order.
%       MP_PJACOBI can compute tiny eigenvalues with high relative accuracy
%       if A is symmetric positive definite. Even if A is ill--conditioned,
%       the MP_PJACOBI can still provide more digits of accuracy than eig. 
%       The algorithm ONLY works for DOUBLE PRECISION input, as the
%       algorithm requires a low precision spectral decomposition as a
%       preconditioner. We simulate the high precision using the Advanpix
%       Multiprecision Toolbox.
%		
%
%   Input:
%    - A is REAL matrix, dimension (N,N)
%       A is a SYMMETRIC, REAL matrix. The function will terminate if the
%       input is NONSYMMETRIC or Complex.
%	 - METHOD is a string
%		Default is "mp3", which indicates using high precision to
%		precondition the Jacobi algorithm.
%		The choice "mp2" uses working precision to precondition. 
%
%   Output:
%    - D is REAL vector, dimension (N)
%       If INFO = 0, D(i) is the ith largest eigenvalue of A.
%
%    - V is REAL matrix, dimension (N,N)
%       If INFO = 0 and EVEC = 1, V(:,i) is an normalized eigenvector 
%       corresponding to D(i). I.e. norm(V'*V - eye(N),2) = macheps, where 
%       macheps is the working precision unit roundoff. 
%       If INFO = 0 and EVEC = 0, V is an identity matrix. 
%   
%    - NROT is INTEGER 
%       The number of applied Jacobi rotation.
%
%    - NSWEEP is INTEGER 
%       The number of sweep the Jacobi rotation required to converge.
%       One sweep indicates the Jacobi algorithm has run through all A(i,j)
%       where i < j.
%
%	The following two outputs are only avaliable if method == "mp3"
%		
%	 - BOUND is DOUBLE
%		The theoretical bound derived in our paper. It is, by definition,
%			7 * n * u * SCOND + sqrt(n) * u
%		where SCOND is the next output.
%
%	 - SCOND is DOUBLE
%		The condition number of preconditioned matrix. Although it is only %		avaliable theoretically, we approximate it by computing the
%		 condition number at high precision. 
%
%   Author:
%       Zhengbo Zhou, June 2025, Manchester, UK
   
% Require eigenvectors 
EVEC = 1;

% Scale?
isscale = 0; 

% MP3 or MP2?
if nargin == 1 
    method = "mp3";
end

% Scale the matrix to avoid overflow 
isscale1 = 0;
anrm1 = max(max(abs(A)));
[eps1, ~, safmin1, ~, ~, ~, ~] = float_params('s'); 
smlnum1 = safmin1 / eps1; 
bignum1 = 1 / smlnum1; 
rmin1 = sqrt( smlnum1 ); 
rmax1 = sqrt( bignum1 ); 

if ( anrm1 > 0 ) && ( anrm1 < rmin1 ) 
    isscale1 = 1; 
    sigma1 = rmin1 / anrm1; 
elseif ( anrm1 > rmax1 )
    isscale1 = 1; 
    sigma1 = rmax1 / anrm1; 
end

if isscale1 == 1 
    Aprime = sigma1 * A;
else
    Aprime = A; 
end

[Qlow,~] = eig(single(Aprime));

% Construct preconditioner at single precision
[Qt,~] = qr(double(Qlow)); % HHQR


% Compute spectral decomposition
if method == "mp2"

    Atcomp = Qt'*A*Qt;
    Atcomp = (Atcomp + Atcomp')/2;
    [QJ,D,NROT,NSWEEP,INFO] = cjacobi(Atcomp,EVEC);

elseif method == "mp3"
    
    % Apply the preconditioned matrix at high precision
    Athcomp = mp(Qt',34) * mp(A,34) * mp(Qt,34);
    
    % Scale the matrix to avoid overflow 
    anrm = max(max(abs(Athcomp)));
    [eps, ~, safmin, ~, ~, ~, ~] = float_params('d'); 
    smlnum = safmin / eps; 
    bignum = 1 / smlnum; 
    rmin = sqrt( smlnum ); 
    rmax = sqrt( bignum ); 

    % In this part, one should notice, it will ensure the overflow does not
    % happened.
    
    if ( anrm > 0 ) && ( anrm < rmin ) 
        isscale = 1; 
        sigma = rmin / anrm; 
    elseif ( anrm > rmax )
        isscale = 1; 
        sigma = rmax / anrm; 
    end
    
    if isscale == 1 
        Athcomp = sigma * Athcomp; 
    end

    % Check the number of zeros to detect underflow 
    tmp1 = nnz(Athcomp); 
    Atcomp = double(Athcomp);
    tmp2 = nnz(Athcomp); 
    if tmp1 ~= tmp2 
        warning("Underflow occurs during demote from high to low precision");
    end

    % Ensure symmetry 
    Atcomp = (Atcomp + Atcomp')/2;

    % Apply Jacobi 
    [QJ,D,NROT,NSWEEP,INFO] = cjacobi(Atcomp,EVEC);
end

% Warn the user that the Jacobi algorithm does not converge
if (INFO == -1)
    warning("Routine cjacobi does not converge");
end

% Further information is required 
if ( nargout >= 5 ) && ( method == "mp3" )
    n = size(A,1);
    u = 2^(-53);
    p1 = sqrt(n); p2 = 7*n;
    SCOND = scond(Athcomp,'D');
    BOUND = p2*u*SCOND+p1*u;
end

% Construct eigenvector
V = Qt * QJ;

% Sort eigenvalues in descending order 
[~,ind] = sort(D,'descend');
D = D(ind); 
V = V(:,ind);

% Scale the eigenvalues back
if isscale == 1
    D = D/sigma; 
end
end

