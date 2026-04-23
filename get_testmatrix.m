function A = get_testmatrix( id )
if id == 1 
    A = anymatrix('nessie/whiskycorr');
elseif id == 2
    % A = anymatrix("gallery/kahan", 50, 1e-2);
    A = gallery('kms',100, 0.5);
elseif id == 3
    % A = gallery('randsvd', 100, -1e8, 3); 
    % A = gallery('kms',100, 0.5);
    A=gallery('lehmer',5e2);
end
end