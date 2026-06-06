clc
clear;

% Construct a star graph with m leaf nodes
m = 20;
G = zeros(m+1);
G(1,2:m+1) = 1;
G(2:m+1,1) = 1;

% Compute tau_1, tau_2, and tau_3
[tau_1, tau_2, tau_3] = f_calculation_tau(G);
tau_1 = round(tau_1,4);
tau_2 = round(tau_2,4);
tau_3 = round(tau_3,4);

% Parameters
e = 0.001;        % error rate
delta = 0.01;     % selection strength
b = 3;            % benefit
c = 1;            % cost

% Transition matrix initialization
trans = zeros(4);
N = m + 1;        % population size

cont1 = 0;        % index counter for states

for i = 1:2
    p = (i-1)*e + (2-i)*(1-e);
    for j = 1:2
        q = (j-1)*e + (2-j)*(1-e);

        s11 = q*(c - b)/(p - q - 1);

        cont1 = cont1 + 1;
        cont2 = 0;

        for h = 1:2
            f = (h-1)*e + (2-h)*(1-e);
            for l = 1:2
                cont2 = cont2 + 1;
                g = (l-1)*e + (2-l)*(1-e);

                s10 = (c*g*(p - q) - b*q*(f - g) + c*q - b*g)/((f - g)*(p - q) - 1);
                s01 = (c*q*(f - g) - b*g*(p - q) + c*g - b*q)/((f - g)*(p - q) - 1);
                s00 = g*(c - b)/(f - g - 1);

                % transition probability under weak selection
                trans(cont2,cont1) = 1/N + delta/N * ...
                    ((s10 - s00)*tau_1 + (s01 - s00)*tau_2 + (s11 + s00 - s01 - s10)*tau_3);
            end
        end
    end
end

for i = 1:4
    trans(i,i) = 1 - sum(trans(i,:)) + trans(i,i);
end

fprintf('transition_matrix =\n');
for i = 1:size(trans,1)
    for j = 1:size(trans,1)
        fprintf('%11.5f', trans(i,j));
    end
    fprintf('\n');
end