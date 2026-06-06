function cate = f_network_category(G)

% Evaluate and convert G
G = double(G);

% Compute tau values
[tau_1, tau_2, tau_3] = f_calculation_tau(G);
tau_1 = round(tau_1,4);
tau_2 = round(tau_2,4);
tau_3 = round(tau_3,4);

% Classification
if tau_2 <= 0
    cate = -1;
    return;
end

% tau_2 > 0 from here
if tau_3 > 0
    if tau_2 >= tau_3
        cate = 1;
        return;
    else
        critical_ep = (2*tau_2 - tau_3) / (2*(tau_2 - tau_3));
    end

elseif tau_3 < 0
    critical_ep = -tau_3 / (2*(tau_2 - tau_3));

else  % tau_3 == 0
    cate = NaN;
    return;
end

% Clamp critical_ep to [0, 1/2]
critical_ep = min(max(critical_ep, 0), 1/2);

% Final classification based on critical_ep
if critical_ep == 0
    cate = 1;
elseif critical_ep == 1/2
    cate = -1;
else
    cate = 0;
end