clc
clear;
% Load predefined adjacency matrices (graph_1, graph_2, ..., graph_21)
load('n_5.mat');
% Initialize category results for 21 networks
cate=zeros(1,21);
for i=1:21
    G = ['graph_', num2str(i)];
    G = eval(G);
    % Compute the category (1: strongly prosocial, 0: weakly prosocial, -1: antisocial)
    cate(i) = f_network_category(G);
end
cate
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Create figure for visualizing all 21 networks

figure('unit','centimeters','position',[5 5 14 6]);

% Create a tight 3-by-7 subplot layout
[ha, pos] = tight_subplot(3, 7, [0.01 0.02], [0.13 0.05], [0.06 0.05]);

% Plot each network in the corresponding subplot
for i = 1:21
    axes(ha(i));

    % Retrieve adjacency matrix by variable name
    G = ['graph_', num2str(i)];
    G = eval(G);

    % Ensure matrix is numeric and convert to MATLAB graph object
    G = double(G);
    G = graph(G);

    % Visualize the network according to its category
    if cate(i) == 1
        % Strongly prosocial structure
        f_draw_network_strongly_prosocial(G, 2);

    elseif cate(i) == 0
        % Weakly prosocial structure
        f_draw_network_weakly_prosocial(G, 2);

    elseif cate(i) == -1
        % Antisocial structure
        f_draw_network_antisocial(G, 2);
    end
end
