function responses = compute_response(params, index, features)
    
    N_filters = size(features, 1);
    N_images = size(features, 2);
    
    theta = params(1);
    s = params(2);
    feature_single = features(index, :);
    responses = ones(N_images, 1) .* s;
    responses(feature_single<theta) = -s;
    
end