function [scores, Y_hat] = strong_classifier_verify_realBoost(h_H, index_H, features, N_weak, B)

    N_filters = size(features, 1);
    N_images = size(features, 2);
    
    scores = zeros(N_images, 1);
    Y_hat = ones(N_images, 1);
    
    for i = 1 : N_weak
        responses = compute_response_realBoost(h_H(i, :), index_H(i), features, B);
        scores = scores + responses;
    end
    
    Y_hat(scores<0) = -1;

end