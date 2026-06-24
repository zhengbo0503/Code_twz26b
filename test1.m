%TEST1 - Test Script
%	Singular vectors
%	Varying condition number
%
addpath("evdalgs");
addpath("svdalgs");

%%
clc; clear; close all; rng(1); 

kkappa = logspace(3,15,15);
mm = 1000*ones(1,length(kkappa));
nn = 800*ones(1,length(kkappa));
epsln = eps('double')/2;
mmode = 3:5;
M = 15;

% Preallocate output arrays
N_kappa = length(kkappa);
N_mode = length(mmode);
err_mposj = zeros(N_kappa, N_mode);
err_dgesvj = zeros(N_kappa, N_mode);
err_dgejsv = zeros(N_kappa, N_mode);
err_matlab = zeros(N_kappa, N_mode);
relgap = zeros(N_kappa, N_mode);
bound2 = zeros(N_kappa, N_mode);
number_failed = zeros(N_kappa, N_mode);

for i = 1:2
    mode = mmode(i);
    for j = 1:length(kkappa)
        kappa = kkappa(j);
        m = mm(j);
        n = nn(j);
        A = gallery('randsvd', [m,n], kappa, mode);
        tol1 = sqrt(m*n)*epsln;

        % Our algorithm
        [U1,S1,V1,nos1,scnd] = mposj(A);
        [Vref, Sref] = get_reference(A, 's');
        [err1, relgap1] = compute_error(A, V1, 's', Vref, Sref);
        [err_mposj(j,i), ind] = max(err1);
        relgap(j,i) = relgap1(ind);

        % Compute bound
        bound2(j,i) = tol1*(1+scnd/relgap(j,i));

        % Access how many eigenvector does not satisfies the bound
        bound_tmp = tol1*(1+scnd./relgap1);
        number_failed(j,i) = sum( err1 > bound_tmp );
        disp(number_failed(j,i));
        if number_failed(j,i) ~= 0
            disp("Error exceed predefined bound");
            pause();
        end

        % one-sided Jacobi
        [U2,S2,V2,sva2,work2,info2] = dgesvj_mex(A,'G','U','V',m,eye(n),max(6,m+n));
        if info2 ~= 0
            fprintf("Error: DGESVJ does not converge.\n");
            break;
        end
        [err2, ~ ] = compute_error(A, V2, 's', Vref, Sref);
        err_dgesvj(j,i) = max(err2);

        % preconditioned one-sided Jacobi
        [U3,S3,V3,sva3,work3,iwork3,info3] = dgejsv_mex(A,'C','U','V','R','N','N');
        if info3 ~= 0
            fprintf("Error: DGEJSV does not converge.\n");
            break;
        end
        [err3, ~ ] = compute_error(A, V3, 's', Vref, Sref);
        err_dgejsv(j,i) = max(err3);

        % MATLAB 
        [U4,S4,V4] = svd(A,'econ');
        [err4, ~] = compute_error(A, V4, 's', Vref, Sref);
        err_matlab(j,i) = max(err4);
        fprintf("Finished MODE=%d, KAPPA %d of %d", mode, j, length(kkappa));
    end
end

%%
for i = 3
    mode = mmode(i);
    for j = 1:length(kkappa) % Loop over condition numbers
        kappa = kkappa(j);
        m = mm(j);
        n = nn(j); 
        tol1 = sqrt(m*n)*epsln;
        err_mposj(j,i)  = -Inf;
        err_dgesvj(j,i) = -Inf;
        err_dgejsv(j,i) = -Inf;
        err_matlab(j,i) = -Inf;

        for k = 1:M
            % Generate test matrix
	        A = gallery('randsvd',[m,n],-kappa,mode);
    
            %  Apply eigensolvers
	        [U1,S1,V1,nos1,scnd] = mposj(A);
	        [Vref, Sref] = get_reference(A, 's');
            [err1, relgap1] = compute_error(A, V1, 's', Vref, Sref);
            [tmp, ind] = max(err1);
            err_mposj(j,i) = max(err_mposj(j,i), tmp);
            if err_mposj(j,i) == tmp
                relgap(j,i) = relgap1(ind);
            end
            
            % Compute bound
            bound2(j,i) = (1+scnd/relgap(j,i))*tol1;
    
            % Access how many eigenvector does not satisfies the bound
            bound_tmp = (1+scnd./relgap1)*tol1;
            number_failed(j,i) = sum( err1 > bound_tmp );
            disp(number_failed(j,i));
            if number_failed(j,i) ~= 0
                disp("Error exceed predefined bound");
                pause();
            end
 
            % one-sided Jacobi
            [U2,S2,V2,sva2,work2,info2] = dgesvj_mex(A,'G','U','V',m,eye(n),max(6,m+n));
            if info2 ~= 0
                fprintf("Error: DGESVJ does not converge.\n");
                break;
            end
            [err2, ~ ] = compute_error(A, V2, 's', Vref, Sref);
            err_dgesvj(j,i) = max(max(err2),err_dgesvj(j,i)); 
    
            % Preconditioned one-sided Jacobi
            [U3,S3,V3,sva3,work3,iwork3,info3] = dgejsv_mex(A,'C','U','V','R','N','N');
            if info3 ~= 0
                fprintf("Error: DGEJSV does not converge.\n");
                break;
            end
            [err3, ~ ] = compute_error(A, V3, 's', Vref, Sref);
            err_dgejsv(j,i) = max(max(err3),err_dgejsv(j,i));
    
            % MATLAB
            [U4,S4,V4] = svd(A,'econ');
            [err4, ~] = compute_error(A, V4, 's', Vref, Sref);
            err_matlab(j,i) = max(max(err4),err_matlab(j,i));
            fprintf("M = %d\n", k); 
        end

        % Print info 
        fprintf("Finished MODE=%d, kappa is %d of %d\n", mode, j, length(kkappa));
    end
end

%%
savedata = 0;
if savedata
    save("data/test1.mat");
end

%%
drawgraph= 1;

if drawgraph
    close all;
    C1 = "#1171BE";
    C2 = "#DD5400";
    C3 = "#EDB120";
    C4 = "#3BAA32";
    
    for mode = 1:3
        figure(mode)
        loglog(kkappa,err_mposj(:,mode),'LineStyle','none','Marker','*', 'Color', C1);
        hold on;
        loglog(kkappa,err_dgejsv(:,mode),'LineStyle','none','Marker','pentagram', 'Color', C2);
        loglog(kkappa,err_dgesvj(:,mode),'LineStyle','none','Marker','square', 'Color', C3);
        loglog(kkappa,err_matlab(:,mode),'LineStyle','none','Marker','diamond', 'Color', C4);
    
        % loglog(kkappa,bound1(:,mode),'LineStyle','--','Marker', 'none', 'Color','k');
        loglog(kkappa,bound2(:,mode),'LineStyle',':','Marker', 'none', 'Color','k');
        set(findall(gcf, 'Type', 'Line'), 'LineWidth', 1);
        
        
        xlabel('Condition number \kappa_2(A)','FontSize',10);
        xlim([1e3,1e15]);
        xticks([1e3,1e6,1e9,1e12,1e15]);
        
        if mod(mode,2) == 1
            ylabel("max svec err");
        end
    
        alphabet = ['a','b','c'];
        t = sprintf('(%s) MODE = %d', alphabet(mode), mode+2); 
        title(t, 'FontWeight', 'normal', 'FontSize', 10); 
    end
end


%%
printout = 0;
if printout
    cleanfigure;
    % EDIT THIS PATH to your own output directory
    matlab2tikz('tmp.tex');
end
