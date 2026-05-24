function nrm = scond(A, type, nrmtype)
%SCOND_TMP - Vectorized version of scond (column scaling path optimized)
	if nargin == 1
	    if (issymmetric(A))
	        type = 'D';
	        nrmtype = 2;
	    else
	        type = 'C';
	        nrmtype = 2;
	    end
	elseif nargin == 2
	    nrmtype = 2;
	end

	switch type
	  case {'C', 'Column', 'column'}
	    D1 = eye(size(A,1));
	    col_norms = sqrt(sum(A.^2, 1));
	    D2 = diag(1 ./ col_norms);
	  case {'D', 'Diagonal', 'diagonal'}
	    D1 = diag(diag(A).^(-1/2));
	    D2 = D1;
	  otherwise
	    error("Invalid scaling type, either 'D' (diagonal scaling) or 'C' (column scaling).");
	end

	nrm = cond(D1*A*D2, nrmtype);
end
