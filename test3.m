%TEST3 - Test Script
%	Eigenvectors
%	Varying condition number
%
addpath("evdalgs");
addpath("svdalgs");
%%
clc; clear; close all; rng(1);

% Parameters 
u = eps('double')/2;
kkappa = logspace(3,15,15);
nn = 500*ones(1,length(kkappa));
mmode = [3,4,5];
M = 15;

% Preallocate output arrays
N_kappa = length(kkappa);
N_mode = length(mmode);
err_mpjacobi = zeros(N_kappa, N_mode);
err_jacobi = zeros(N_kappa, N_mode);
err_pjacobi = zeros(N_kappa, N_mode);
err_matlab = zeros(N_kappa, N_mode);
relgap = zeros(N_kappa, N_mode);
bound2 = zeros(N_kappa, N_mode);
number_failed = zeros(N_kappa, N_mode);

for i = 1:2
    % Main
    mode = mmode(i);

    for j = 1:length(kkappa) % Loop over condition numbers
        n = nn(j); 
        kappa = kkappa(j);
        tol1 = 3*sqrt(n)*u;

        % Generate test matrix
	    A = gallery('randsvd',n,-kappa,mode);

        %  Apply eigensolvers
	    [V1,D1,nrot1,nsweep1,xbound1,scnd] = mp_pjacobi(A,'mp3');
	    [Vref, Sref] = get_reference(A, 'E');
        [err1, relgap1] = compute_error(A, V1, 'E', Vref, Sref);
        [err_mpjacobi(j,i), ind] = max(err1);
        relgap(j,i) = relgap1(ind);
        
        % Compute bound
        bound2(j,i) = tol1*(1+scnd/relgap(j,i));

        % Access how many eigenvector does not satisfies the bound
        bound_tmp = tol1*(1+scnd./relgap1);
        number_failed(j,i) = sum( err1 > bound_tmp );
        fprintf("Number of failed = %d\n", number_failed(j,i));
        if number_failed(j,i) ~= 0
            beep;
            pause();
        end

        % Standard two-sided Jacobi 
        [V2,D2] = cjacobi(A);
        [err2, ~ ] = compute_error(A, V2, 'E', Vref, Sref);
        err_jacobi(j,i) = max(err2);

        % Preconditioned two-sided Jacobi
        [V3,D3] = mp_pjacobi(A,'mp2');
        [err3, ~ ] = compute_error(A, V3, 'E', Vref, Sref);
        err_pjacobi(j,i) = max(err3);

        % MATLAB
        [V4,D4] = eigsort(A);    
        [err4, ~ ] = compute_error(A, V4, 'E', Vref, Sref);
        err_matlab(j,i) = max(err4);

        % Print info 
        fprintf("----Finished MODE=%d, KAPPA %d of %d\n", mode, j, length(kkappa));
    end
end

%%
for i = 3
    mode = mmode(i);
    for j = 1:length(kkappa) % Loop over condition numbers
        kappa = kkappa(j);
        n = nn(j); 
        tol1 = 3*sqrt(n)*u;
        err_mpjacobi(j,i) = -Inf;
        err_jacobi(j,i) = -Inf;
        err_pjacobi(j,i) = -Inf;
        err_matlab(j,i) = -Inf;

        for k = 1:M
            % Generate test matrix
	        A = gallery('randsvd',n,-kappa,mode);
    
            %  Apply eigensolvers
	        [V1,D1,~,~,~,scnd] = mp_pjacobi(A,'mp3');
	        [Vref, Sref] = get_reference(A, 'E');
            [err1, relgap1] = compute_error(A, V1, 'E', Vref, Sref);
            [tmp, ind] = max(err1);
            err_mpjacobi(j,i) = max(err_mpjacobi(j,i), tmp);
            if err_mpjacobi(j,i) == tmp
                relgap(j,i) = relgap1(ind);
            end
            
            % Compute bound
            bound2(j,i) = (1+scnd/relgap(j,i))*tol1;
    
            % Access how many eigenvector does not satisfies the bound
            bound_tmp = (1+scnd./relgap1)*tol1;
            number_failed(j,i) = sum( err1 > bound_tmp );
            disp(number_failed(j,i));
            if number_failed(j,i) ~= 0
                beep;
                pause();
            end
    
            % Standard two-sided Jacobi 
            [V2,D2] = cjacobi(A);
            [err2, ~ ] = compute_error(A, V2, 'E', Vref, Sref);
            err_jacobi(j,i) = max(max(err2),err_jacobi(j,i));
    
            % Preconditioned two-sided Jacobi
            [V3,D3] = mp_pjacobi(A,'mp2');
            [err3, ~ ] = compute_error(A, V3, 'E', Vref, Sref);
            err_pjacobi(j,i) = max(max(err3),err_pjacobi(j,i));
    
            % MATLAB
            [V4,D4] = eigsort(A);    
            [err4, ~ ] = compute_error(A, V4, 'E', Vref, Sref);
            err_matlab(j,i) = max(max(err4),err_matlab(j,i));
        end

        % Print info 
        fprintf("Finished MODE=%d, N is %d of %d\n", mode, j, length(nn));
    end
end


beep;

%%
savedata = 0;
if savedata
    save("data/test3.mat");
end

%%
drawgraph=1;
if drawgraph
    close all;
    C1 = "#1171BE";
    C2 = "#DD5400";
    C3 = "#EDB120";
    C4 = "#3BAA32";
    
    for mode = 1:3
        figure(mode)
        loglog(kkappa,err_mpjacobi(:,mode),'LineStyle','none','Marker','*', 'Color', C1, 'MarkerSize', 25, 'LineWidth', 1);
        hold on;
        loglog(kkappa,err_jacobi(:,mode),'LineStyle','none','Marker','pentagram', 'Color', C2, 'MarkerSize', 25, 'LineWidth', 1);
        loglog(kkappa,err_pjacobi(:,mode),'LineStyle','none','Marker','square', 'Color', C3, 'MarkerSize', 25, 'LineWidth', 1);
        loglog(kkappa,err_matlab(:,mode),'LineStyle','none','Marker','diamond', 'Color', C4, 'MarkerSize', 25, 'LineWidth', 1);
        loglog(kkappa,bound2(:,mode),'LineStyle',':','Marker', 'o', 'Color','k','LineWidth',2);
        
        
        
        xlabel('Condition number \kappa_2(A)','FontSize',10);
        xlim([1e3,1e15]);
        xticks([1e3,1e6,1e9,1e12,1e15]);
        
        if mod(mode,2) == 1
            ylabel("max evec err");
        end
    
        if mode == 3
            legend('MP3Jacobi', 'Jacobi', 'MP2Jacobi', 'MATLAB');
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
    matlab2tikz('/Users/cyae/Dropbox/tex/mp_jacobi_evecs_svecs/figs/tmp.tex');
end
