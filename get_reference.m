function [Vmp, S] = get_reference(A, Vtype)
%GET_REFERENCE Compute exact singular/eigen-vectors using multiprecision arithmetic.
%   [Vmp, S] = GET_REFERENCE(A, Vtype) returns the reference right singular
%   vectors (Vtype='S'), left singular vectors (Vtype='L'), or eigenvectors
%   (Vtype='E') and the corresponding singular/eigen-values S, computed in
%   34-digit precision via Advanpix mp.
%
%   Use with compute_error(..., Vmp, S) to avoid recomputing the reference
%   when comparing multiple algorithms on the same matrix.

mp.Digits(34);
Amp = mp(A);

switch Vtype
  case {'S', 's'}
    [~, Smp, Vmp] = svd(Amp, 'econ');
    S = diag(Smp);
  case {'L', 'l'}
    [Ump, Smp, ~] = svd(Amp, 'econ');
    S = diag(Smp);
    Vmp = Ump;
  case {'E', 'e'}
    [Vmp, Dmp] = eig(Amp);
    S = diag(Dmp);
    [S, ind] = sort(S, 'descend');
    Vmp = Vmp(:, ind);
  otherwise
    error("Invalid Vtype; use 'S' for singular vectors, 'L' for left singular vectors, or 'E' for eigenvectors.");
end
end
