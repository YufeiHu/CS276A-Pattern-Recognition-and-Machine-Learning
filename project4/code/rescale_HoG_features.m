function [features_rescale] = rescale_HoG_features(features)
    
    min_HoG = min(min(features));
    max_HoG = max(max(features));
    
    features_rescale = (features - min_HoG) ./ (max_HoG - min_HoG);

end