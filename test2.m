%TEST2 -- access the accuracy of the computed singular vectors
%		with varying number of columns.

addpath("evdalgs");
addpath("svdalgs");
%%
clc; close all; clear; rng(1);

mm = ones(15,1)*1000;
nn = round(logspace(1,3,15));
epsln = eps('double')/2;
kappa = 1e8;

mmode = 3:5;

for i = 1:length(mmode)
    mode = mmode(i);
    for j = 1:length(nn)
		m = mm(j); 
        n = nn(j);
        A = gallery('randsvd', [m,n], kappa, mode);
        tol1 = sqrt(m*n)*epsln;

        % Our algorithm
        [U1,S1,V1,nos1,scnd] = mposj(A);
        [err1, relgap1] = compute_error(A,V1,'s');
        [err_mposj(j,i), ind] = max(err1);
        relgap(j,i) = relgap1(ind);

        % Compute bound
        bound2(j,i) = tol1*(1+scnd/relgap(j,i));

        % Access how many eigenvector does not satisfies the bound
        bound_tmp = tol1*(1+scnd./relgap1);
        number_failed(j,i) = sum( err1 > bound_tmp );
        disp(number_failed(j,i));

        % one-sided Jacobi
        [U2,S2,V2,sva2,work2,info2] = dgesvj_mex(A,'G','U','V',m,eye(n),max(6,m+n));
        if info2 ~= 0
            fprintf("Error: DGESVJ does not converge.\n");
            break;
        end
        [err2, ~ ] = compute_error(A,V2,'s');
        err_dgesvj(j,i) = max(err2);

        % preconditioned one-sided Jacobi
        [U3,S3,V3,sva3,work3,iwork3,info3] = dgejsv_mex(A,'C','U','V','R','N','N');
        if info3 ~= 0
            fprintf("Error: DGEJSV does not converge.\n");
            break;
        end
        [err3, ~ ] = compute_error(A,V3,'s');
        err_dgejsv(j,i) = max(err3);

        % MATLAB 
        [U4,S4,V4] = svd(A,'econ');
        [err4, ~] = compute_error(A,V4,'s');
        err_matlab(j,i) = max(err4);
        fprintf("Finished MODE=%d, n %d of %d\n", mode, j, length(nn));
    end
end

%%
savedata = 0;
if savedata
    save("data/test2.mat");
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
        loglog(nn,err_mposj(:,mode),'LineStyle','none','Marker','*', 'Color', C1);
        hold on;
        loglog(nn,err_dgejsv(:,mode),'LineStyle','none','Marker','pentagram', 'Color', C2);
        loglog(nn,err_dgesvj(:,mode),'LineStyle','none','Marker','square', 'Color', C3);
        loglog(nn,err_matlab(:,mode),'LineStyle','none','Marker','diamond', 'Color', C4);
    
        loglog(nn,bound2(:,mode),'LineStyle',':','Marker', 'none', 'Color','k');
        set(findall(gcf, 'Type', 'Line'), 'LineWidth', 1);
        
        xlabel('$n$','FontSize',10);
        xlim([1e1,1e3]);
        % xticks([1e3,1e6,1e9,1e12,1e15]);
        
        if mod(mode,2) == 1
            ylabel("max svec err");
        end
    
        alphabet = ['a','b','c'];
        t = sprintf('(%s) MODE = %d', alphabet(mode), mode+2); 
        title(t, 'FontWeight', 'normal', 'FontSize', 10); 
    end
end

%%  
printout = 1;
if printout 
    cleanfigure;
    matlab2tikz('/Users/cyae/Dropbox/tex/mp_jacobi_evecs_svecs/figs/tmp.tex');
end