%TEST4 - Test Script
%	Eigenvectors
%	Varying matrix size
%
addpath("evdalgs");
addpath("svdalgs");
%%
clc; clear; close all; rng(1);

% Parameters 
nn = round(logspace(1,3,15));
u = eps('double')/2;
kkappa = 1e8*ones(1,length(nn));
mmode = [3,4,5];
M = 15;

% Preallocate output arrays
N_n = length(nn);
N_mode = length(mmode);
err_mpjacobi = zeros(N_n, N_mode);
err_jacobi = zeros(N_n, N_mode);
err_pjacobi = zeros(N_n, N_mode);
err_matlab = zeros(N_n, N_mode);
relgap = zeros(N_n, N_mode);
bound2 = zeros(N_n, N_mode);
number_failed = zeros(N_n, N_mode);

for i = 1:2
    % Main
    mode = mmode(i);

    for j = 1:length(nn) % Loop over different dimensions
        kappa = kkappa(1);
        n = nn(j); 
        tol1 = 3*sqrt(n)*u;

        % Generate test matrix
	    A = gallery('randsvd',n,-kappa,mode);

        %  Apply eigensolvers
	    [V1,D1,~,~,~,scnd] = mp_pjacobi(A,'mp3');
	    [Vref, Sref] = get_reference(A, 'E');
        [err1, relgap1] = compute_error(A, V1, 'E', Vref, Sref);
        [err_mpjacobi(j,i), ind] = max(err1);
        relgap(j,i) = relgap1(ind);
        
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
        fprintf("Finished MODE=%d, N is %d of %d\n", mode, j, length(nn));
    end
end

%%
for i = 3
    mode = mmode(i);
    for j = 1:length(nn) % Loop over condition numbers
        kappa = kkappa(1);
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
                disp("Error exceed predefined bound");
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

%%

savedata = 0;
if savedata
    save("data/test4.mat");
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
        loglog(nn,err_mpjacobi(:,mode),'LineStyle','none','Marker','*', 'Color', C1, 'MarkerSize', 25, 'LineWidth', 1);
        hold on;
        loglog(nn,err_jacobi(:,mode),'LineStyle','none','Marker','pentagram', 'Color', C2, 'MarkerSize', 25, 'LineWidth', 1);
        loglog(nn,err_pjacobi(:,mode),'LineStyle','none','Marker','square', 'Color', C3, 'MarkerSize', 25, 'LineWidth', 1);
        loglog(nn,err_matlab(:,mode),'LineStyle','none','Marker','diamond', 'Color', C4, 'MarkerSize', 25, 'LineWidth', 1);
        loglog(nn,bound2(:,mode),'LineStyle',':','Marker', 'o', 'Color','k','LineWidth',2);
        
        xlabel('n','FontSize',10);
        xlim([1e1,1e3]);
        
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
    matlab2tikz('tmp.tex');
end
