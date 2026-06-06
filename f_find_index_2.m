function index = f_find_index_2(seq, n)
seq = sort(seq);
x = seq(1);
y = seq(2);
index = (x - 1) * n + y - (x + 1) * x / 2;
