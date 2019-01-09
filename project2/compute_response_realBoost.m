function responses = compute_response_realBoost(h_optimal, i_optimal, features_train, B)
    
    N_filters = size(features_train, 1);
    N_images = size(features_train, 2);
    
    feature_single = features_train(i_optimal, :);
    h_pool = linspace(min(feature_single), max(feature_single), B+1);
    
    responses = zeros(N_images, 1);
    for j = 1 : N_images
        b_index = bin_index_count(feature_single(j), h_pool);
        responses(j) = h_optimal(b_index);
    end
    
end