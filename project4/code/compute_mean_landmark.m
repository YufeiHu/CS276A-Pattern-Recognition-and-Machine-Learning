function [ML] = compute_mean_landmark(features)

    num_LM = size(features, 2) / 2;
    ML = zeros( num_LM * 2, 1 );
    j = 1;
    for i = 1 : size(features, 1)
        ML = ML + features(i, :)';
        j = j + 1;
    end
    ML = ML ./ size(features, 1);

end