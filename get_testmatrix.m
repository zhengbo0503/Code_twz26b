function A = get_testmatrix( id )
if id == 1
    A = gallery('kms',100, 0.5);
elseif id == 2
    A = gallery('lehmer',5e2);
end
end