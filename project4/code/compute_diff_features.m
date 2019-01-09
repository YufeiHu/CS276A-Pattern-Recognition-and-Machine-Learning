function [gov_features] = compute_diff_features(gov_landmark_rescale)

    num_pair = size(gov_landmark_rescale, 1) / 2;
    gov_features = zeros(num_pair, size(gov_landmark_rescale, 2));
    
    for i = 1 : num_pair
        gov_features(i, :) = gov_landmark_rescale(2*i-1, :) - gov_landmark_rescale(2*i, :);
    end

end