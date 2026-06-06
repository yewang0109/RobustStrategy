function f_draw_network_strongly_prosocial(Gr, size)
    % Set random seed to ensure reproducibility
    rng(0); 
    % Plot the network
    G = plot(Gr, 'Layout', 'auto');
    % Set node size
    G.MarkerSize = size;
    % Set edge line width
    G.LineWidth = 0.5;
    % Set edge color
    G.EdgeColor = [181, 181, 182]./255;
    % Set edge transparency (50%)
    G.EdgeAlpha = 0.5;
    % Set node color
    G.NodeColor = [131, 203, 172]./255;
    % Remove node labels
    G.NodeLabel = '';
    % Get node coordinates
    node_coords = get(G, 'XData')';
    node_coords = [node_coords; get(G, 'YData')'];
    % Check whether node_coords is empty
    if isempty(node_coords)
        error('No node coordinates found.');
    end
end