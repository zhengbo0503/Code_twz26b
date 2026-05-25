% =========================================================
% Compute absolute and relative eigenvalue gaps
% for SPD gallery("randsvd", n, -kappa, mode)
%
% Definitions:
%   absgap(lambda_i) = min_{j ~= i} |lambda_i - lambda_j| / ||H||_2
%
%   relgap(lambda_i) = min_{j ~= i} |lambda_i - lambda_j|
%                      / sqrt(|lambda_i * lambda_j|)
% =========================================================

clear; clc;

% -----------------------------
% User inputs
% -----------------------------
n     = 800;
mode  = 5;
kappa = 1e8;

if kappa < 1
    error('kappa must be >= 1.');
end

% -----------------------------
% Generate SPD test matrix
% -----------------------------
% For gallery("randsvd"), using negative kappa produces
% a symmetric positive definite matrix with condition number kappa.
H = gallery('randsvd', n, -kappa, mode);

% -----------------------------
% Eigenvalues
% -----------------------------
lambda = eig(H);
lambda = sort(lambda, 'descend');

normH = norm(H, 2);

% -----------------------------
% Pairwise absolute differences
% -----------------------------
D = abs(lambda - lambda.');

% Exclude j = i from the minimum
D(1:n+1:end) = inf;

% -----------------------------
% Absolute gap
% absgap_i = min_{j ~= i} |lambda_i - lambda_j| / ||H||_2
% -----------------------------
absgap = min(D / normH, [], 2);

% -----------------------------
% Relative gap
% relgap_i = min_{j ~= i} |lambda_i - lambda_j| / sqrt(|lambda_i lambda_j|)
% -----------------------------
denom = sqrt(abs(lambda * lambda.'));

R = D ./ denom;
R(1:n+1:end) = inf;

relgap = min(R, [], 2);

% -----------------------------
% Output table
% -----------------------------
GapTable = table((1:n).', lambda, absgap, relgap, ...
    'VariableNames', {'Index', 'Eigenvalue', 'AbsGap', 'RelGap'});

disp(GapTable);