function [Vmp, S] = get_reference(A, Vtype)
%GET_REFERENCE Compute exact singular/eigen-vectors using multiprecision arithmetic.
%   [Vmp, S] = GET_REFERENCE(A, Vtype) returns the reference right singular
%   vectors (Vtype='S') or eigenvectors (Vtype='E') and the corresponding
%   singular/eigen-values S, computed in 34-digit precision via Advanpix mp.
%
%   Use with compute_error(..., Vmp, S) to avoid recomputing the reference
%   when comparing multiple algorithms on the same matrix.

mp.Digits(34);
Amp = mp(A);

switch Vtype
  case {'S', 's'}
    [~, Smp, Vmp] = svd(Amp, 'econ');
    S = diag(Smp);
  case {'E', 'e'}
    [Vmp, Dmp] = eig(Amp);
    S = diag(Dmp);
    [S, ind] = sort(S, 'descend');
    Vmp = Vmp(:, ind);
  otherwise
    error("Invalid Vtype; use 'S' for singular vectors or 'E' for eigenvectors.");
end
end
