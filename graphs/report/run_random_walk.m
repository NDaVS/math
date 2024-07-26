

args = argv();
if numel(args) < 3
        error('Необходимо указать имя файла с матрицей смежности и',
                'номера двух узлов.');
end

file_path = args{1};
start = str2double(args{2});
end_ = str2double(args{3});
num_sim = 1000;

adj = dlmread(file_path);


tic;
[fht, ct, mfht, mct, eff_res, mcmt] = random_walk(adj, start, end_, num_sim);
elapsed_time = toc;


fprintf('Среднее время первого попадания в вершину %d из' ,
        'вершины %d: %f шага.\n', end_, start, mfht);
fprintf('Среднее время прохода из вершины %d в вершину %d и' , 
        'обратно: %f шага.\n', start, end_, mcmt);
fprintf('Среднее время обхода всего графа: %f шага.\n', mct);
fprintf('Сопротивление: %f.\n', eff_res);
fprintf('Время выполнения программы: %f секунд.\n', elapsed_time);

