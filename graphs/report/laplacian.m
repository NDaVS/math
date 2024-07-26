
laplacian_matrix = [2 -1 -1 0;                
                   -1 3 -1 -1;
                   -1 -1 3 -1;
                    0 -1 -1 2];


[eigenvectors, eigenvalues_matrix] = eig(laplacian_matrix);

eigenvalues = diag(eigenvalues_matrix);

disp('Собственные значения:');
disp(eigenvalues);

disp('Собственные вектора:');
disp(eigenvectors);
