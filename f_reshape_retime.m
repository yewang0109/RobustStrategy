function [retime2_mat, retime3_mat] = f_reshape_retime(retime2, retime3, n)
retime2_mat = zeros(n, n);
retime3_mat = zeros(n, n, n);

tik = 1;
for i = 1: n
    for j = i: n
        seq = [i, j];
        seq = unique(seq);
        len = length(seq);
        if len == 1
            retime2_mat(i, j) = 0;
            retime2_mat(j, i) = 0;
        else
            retime2_mat(i, j) = retime2(tik, 1);
            retime2_mat(j, i) = retime2(tik, 1);
            tik = tik + 1;
        end
    end
end

tik = 1;
for i = 1: n
    for j = i: n
        for k = j: n
            seq = [i, j, k];
            seq = unique(seq);
            len = length(seq);
            if len == 1
                retime3_mat(i, j, k) = 0; retime3_mat(i, k, j) = 0;
                retime3_mat(j, i, k) = 0; retime3_mat(j, k, i) = 0;
                retime3_mat(k, i, j) = 0; retime3_mat(k, j, i) = 0;
            elseif len == 2
                x = seq(1); y = seq(2); val = retime2_mat(x, y);
                retime3_mat(i, j, k) = val; retime3_mat(i, k, j) = val;
                retime3_mat(j, i, k) = val; retime3_mat(j, k, i) = val;
                retime3_mat(k, i, j) = val; retime3_mat(k, j, i) = val;
            else
                val = retime3(tik, 1);
                retime3_mat(i, j, k) = val; retime3_mat(i, k, j) = val;
                retime3_mat(j, i, k) = val; retime3_mat(j, k, i) = val;
                retime3_mat(k, i, j) = val; retime3_mat(k, j, i) = val;
                tik = tik + 1;
            end
        end
    end
end
