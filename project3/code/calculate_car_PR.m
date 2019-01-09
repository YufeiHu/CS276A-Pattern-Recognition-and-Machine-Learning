function [cur_PR] = calculate_car_PR(cur_bounding_boxes_picked, cur_scores_picked, true_bbox)
    
    threshold_overlap = 0.5;
    
    if(isempty(cur_bounding_boxes_picked))
        cur_PR = [];
    elseif(isempty(true_bbox))
        cur_PR = [zeros(size(cur_scores_picked)) cur_scores_picked];
    else
        cur_PR = [zeros(size(cur_scores_picked)) cur_scores_picked];
        
        for i = 1 : size(cur_bounding_boxes_picked, 1)
            
            cur_pick = cur_bounding_boxes_picked(i, :);
            for j = 1 : size(true_bbox, 1)
                cur_true = true_bbox(j, :);
                overlapArea = rectint(cur_true, cur_pick);
                area_true = cur_true(3) * cur_true(4);
                area_pick = cur_pick(3) * cur_pick(4);
                if( ((overlapArea / area_true) > threshold_overlap) || ((overlapArea / area_pick) > threshold_overlap) )
                    cur_PR(i, 1) = 1;
                    break
                end
            end
            
        end
        
    end

end