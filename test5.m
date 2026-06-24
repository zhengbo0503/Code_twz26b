%TEST5 - Access the accuracy of the computed singular vectors for special
%matrices

clear; close all; clc;
addpath("evdalgs");
addpath("svdalgs");

epsln = eps('double')/2;

for id = 1:2
    A = get_testmatrix(id); 
    [m,n] = size(A); 

    tol1 = sqrt(m*n)*epsln;

    [~,S1,V1,~,scnd] = mposj(A);
    [Vref, Sref] = get_reference(A, 's');
    [err1, relgap1] = compute_error(A, V1, 's', Vref, Sref);

    bound = (1+scnd./relgap1)*tol1;

    [U2,S2,V2,sva2,work2,info2] = dgesvj_mex(A,'G','U','V',m,eye(n),max(6,m+n));
    if info2 ~= 0
        fprintf("Error: DGESVJ does not converge.\n");
        break;
    end
    [err2, ~ ] = compute_error(A, V2, 's', Vref, Sref);
    
    % preconditioned one-sided Jacobi
    [U3,S3,V3,sva3,work3,iwork3,info3] = dgejsv_mex(A,'C','U','V','R','N','N');
    if info3 ~= 0
        fprintf("Error: DGEJSV does not converge.\n");
        break;
    end
    [err3, ~ ] = compute_error(A, V3, 's', Vref, Sref);
    
    % MATLAB 
    [U4,S4,V4] = svd(A,'econ');
    [err4, ~] = compute_error(A, V4, 's', Vref, Sref);

    figure(id); 
    C1 = "#1171BE";
    C2 = "#DD5400";
    C3 = "#EDB120";
    C4 = "#3BAA32";
    if id == 2


        semilogy(1:10:n,err1(1:10:n),'LineStyle','none','Marker','*', 'Color', C1);
        hold on;
        semilogy(1:10:n,err2(1:10:n),'LineStyle','none','Marker','pentagram', 'Color', C2);
        semilogy(1:10:n,err3(1:10:n),'LineStyle','none','Marker','square', 'Color', C3);
        semilogy(1:10:n,err4(1:10:n),'LineStyle','none','Marker','diamond', 'Color', C4);
        
        semilogy(1:10:n,bound(1:10:n),'LineStyle',':','Marker', 'none', 'Color','k');
    else
          
        semilogy(1:n,err1,'LineStyle','none','Marker','*', 'Color', C1);
        hold on;
        semilogy(1:n,err2,'LineStyle','none','Marker','pentagram', 'Color', C2);
        semilogy(1:n,err3,'LineStyle','none','Marker','square', 'Color', C3);
        semilogy(1:n,err4,'LineStyle','none','Marker','diamond', 'Color', C4);
        
        semilogy(1:n,bound,'LineStyle',':','Marker', 'none', 'Color','k');

    end
    set(findall(gcf, 'Type', 'Line'), 'LineWidth', 1);

    xlabel('Singular vector index (ordered by descending singular value)')
    ylabel('max svec err')
    
    legend('MP3JacobiSVD', 'DGESVJ', 'DGEJSV', 'MATLAB svd'); 
        
end

%%
printout = 0;
if printout 
    cleanfigure;
    % EDIT THIS PATH to your own output directory
    matlab2tikz('tmp.tex');
end