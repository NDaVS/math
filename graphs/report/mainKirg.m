

args = argv();
if numel(args) < 3
        error('Необходимо указать имя файла с матрицей смежности и', 
                'номера двух узлов.');
end
filename = args{1};

a = str2double(args{2});
b = str2double(args{3});
if isnan(a) || isnan(b) || a <= 0 || b <= 0 
        error('Номера узлов должны быть положительными целыми ',
                'числами.');
endif

tic;
[~, name, ~] = fileparts(filename);
pseudo_inverse_filename = [name, '_L_plus.mat'];

if exist(pseudo_inverse_filename, 'file') == 2
        load(pseudo_inverse_filename, 'L_plus');
        disp(['Псевдообратная матрица Лапласа загружена из файла ', 
                pseudo_inverse_filename, '.']);
        n = size(L_plus, 1);


else
        A = dlmread(filename);
        n = size(A, 1);

        
        D = diag(sum(A, 2));
        L = D - A;
        if rank(L) < n-1
                error('Матрица Лапласиана не имеет полного ранга.');
        end
        L_plus = pinv(L);
        save(pseudo_inverse_filename, 'L_plus');
        disp(['Псевдообратная матрица Лапласа вычислена и ',
                'сохранена в файл ', pseudo_inverse_filename, '.']);
end


if a > n || b > n
        error('Номера узлов выходят за пределы размерности ',
                'матрицы.');
endif

answer = (L_plus(a, a) - L_plus(b, a)) - (L_plus(a, b) - 
        L_plus(b, b));


elapsed_time = toc;
disp("===========================================================");
disp(['Резистивное расстояние между узлами ', num2str(a), ' и ',
         num2str(b), ': ', num2str(answer)]);
disp(['Время выполнения программы: ', num2str(elapsed_time), 
        ' секунд']);

