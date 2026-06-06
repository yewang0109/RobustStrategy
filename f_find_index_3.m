function index = f_find_index_3(seq, n)
seq = sort(seq);
i = seq(1);
j = seq(2);
k = seq(3);
val = 0;
for a = 1: i-1
    val = val + nchoosek(n-a, 2);
end
for b = i+1: j-1
    val = val + (n-b);
end
index = val + (k - j);