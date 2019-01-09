function [features_rescale] = rescale_features(features)
    

    landmark_pos_offset = size(features, 2) / 2;
    
    min_x = min(min(features(:, 1:landmark_pos_offset)));
    max_x = max(max(features(:, 1:landmark_pos_offset)));
    min_y = min(min(features(:, landmark_pos_offset+1:end)));
    max_y = max(max(features(:, landmark_pos_offset+1:end)));
    
    features_rescale = zeros(size(features));
    
    features_rescale(:, 1:landmark_pos_offset) = (features(:, 1:landmark_pos_offset) - min_x) ./ (max_x - min_x);
    features_rescale(:, 1+landmark_pos_offset:end) = (features(:, 1+landmark_pos_offset:end) - min_y) ./ (max_y - min_y);
    
%     
%     threshold_min = 0.1;
%     threshold_max = 0.9;
%     ncol_ori = size(features, 2);
%     features_rescale = zeros(size(features));
%     
%     for i = 1 : ncol_ori
%         cur_vec = features(:, i);
%         cur_min = min(cur_vec);
%         cur_max = max(cur_vec);
%         cur_rescale = (cur_vec - cur_min) ./ (cur_max - cur_min);
%         cur_rescale(cur_rescale<threshold_min) = threshold_min;
%         cur_rescale(cur_rescale>threshold_max) = threshold_max;
%     	features_rescale(:, i) = cur_rescale;
%     end

end