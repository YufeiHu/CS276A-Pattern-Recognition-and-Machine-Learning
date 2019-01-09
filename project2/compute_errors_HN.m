function [error_1000, optimal_filter_error, optimal_filter_index, optimal_filter_params] = compute_errors_HN(Y, features, w)

    N_filters = size(features, 1);
    N_images = size(features, 2);
    
    error_optimal = zeros(N_filters, 1);
    filter_params = zeros(N_filters, 2);
    s_pool = [1 -1];
    
    theta_optimal = zeros(N_filters, 1);
    s_params = zeros(N_filters, 1);
    
    parfor i = 1 : N_filters

        feature_single = features(i, :);
        
        theta_pool = linspace(min(feature_single), max(feature_single), 40);
        
        [error, theta, s] = compute_errors_helper_HN(feature_single, s_pool, theta_pool, w', Y');
        
        error_optimal(i, 1) = error;
        theta_optimal(i, 1) = theta;
        s_params(i, 1) = s;
        
    end
    
    filter_params(:, 1) = theta_optimal;
    filter_params(:, 2) = s_params;
    [optimal_filter_error, optimal_filter_index] = min(error_optimal);
    optimal_filter_params = filter_params(optimal_filter_index, :);
    error_sort = sort(error_optimal);
    error_1000 = error_sort(1 : 1000);
    
end