function test_xgesvj
%TEST_XGESVJ is used to test the following mex files.
% - DGESVJ
% - SGESVJ
% - DGEJSV
% - SGEJSV
%	The test is focused on whether they are able to run, not on the accuracy
%	or speed.
A = randn(4,3);
Asingle = single(A);
[m,n] = size(A);
V0 = eye(n);
V0single = single(V0);

[~,~,~,~,~,info] = dgesvj_mex(A,'G','U','V',n,V0,max(6,m+n));
if info ~= 0
    fprintf(" -- DGESVJ failed\n");
    return
else
    fprintf(" -- DGESVJ works fine.\n");
end

[~,~,~,~,~,info] = sgesvj_mex(Asingle,'G','U','V',n,V0single,max(6,m+n));
if info ~= 0
    fprintf(" -- SGESVJ failed\n");
    return
else
    fprintf(" -- SGESVJ works fine.\n");
end

[~,~,~,~,~,~,info] = dgejsv_mex(A,'F','U','V','N','T','N');
if info ~= 0
    fprintf(" -- DGEJSV failed\n");
    return
else
	fprintf(" -- DGEJSV works fine.\n");    
end

[~,~,~,~,~,~,info] = sgejsv_mex(Asingle,'F','U','V','N','T','N');
if info ~= 0
    fprintf(" -- SGEJSV failed\n");
    return
else
    fprintf(" -- SGEJSV works fine.\n");
end

end
