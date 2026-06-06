function [tau_1,tau_2,tau_3]=f_calculation_tau(G)
Gr=full(G);
num_nodes=size(Gr, 1);
madj2 = Gr;
replace_mat = madj2;
repro_val = sum(replace_mat) / sum(sum(replace_mat));
trans_mat = f_gen_trans_mat(replace_mat, num_nodes);
retime2 = f_cal_remeet_time_two(trans_mat, num_nodes);
retime3 = f_cal_remeet_time_three(trans_mat, retime2, num_nodes);
[retime2_mat, retime3_mat] = f_reshape_retime(retime2, retime3, num_nodes); %%%
% matrix form
rowSums = sum(Gr, 2);
G_1=Gr./rowSums;
G_2=G_1*G_1;
G_3=G_1*G_1*G_1;
cont1=0;
cont2=0;
cont3=0;
a=eye(num_nodes);
for i=1:num_nodes
    for j=1:num_nodes
        cont1=cont1+rowSums(i)/(sum(rowSums))*(G_2(i,j)-a(i,j))*retime2_mat(i,j);
        cont2=cont2+rowSums(i)/(sum(rowSums))*(G_3(i,j)-G_1(i,j))*retime2_mat(i,j);
        for y=1:num_nodes
            cont3=cont3+rowSums(i)/(sum(rowSums))*(G_2(i,j)-a(i,j))*G_1(j,y)*retime3_mat(i,j,y);
        end
    end
end
tau_1=cont1;
tau_2=cont2;
tau_3=cont3;
end

