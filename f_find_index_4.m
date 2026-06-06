function index = f_find_index_4(seq, n)
seq = sort(seq);
i = seq(1);
j = seq(2);
k = seq(3);
l = seq(4);
val = 0;
for a = 1: i-1
    val = val + nchoosek(n-a, 3);
end
for b = i+1: j-1
    val = val + nchoosek(n-b, 2);
end
for c = j+1: k-1
    val = val + nchoosek(n-c, 1);
end
index = val + (l - k);