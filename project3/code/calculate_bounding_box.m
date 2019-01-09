function [true_bounding, true_score] = calculate_bounding_box(cur_bounding_boxes_picked, cur_scores_picked, true_bbox)
    
    threshold_overlap = 0.5;
    
    if(isempty(cur_bounding_boxes_picked))
        true_bounding = [];
        true_score = [];
    elseif(isempty(true_bbox))
        true_bounding = [];
        true_score = [];
    else
        true_bounding = [];
        true_score = [];
        for i = 1 : size(true_bbox, 1)
            cur_true = true_bbox(i, :);
            for j = 1 : size(cur_bounding_boxes_picked, 1)
                cur_pick = cur_bounding_boxes_picked(j, :);
                
                overlapArea = rectint(cur_true, cur_pick);
                area_true = cur_true(3) * cur_true(4);
                area_pick = cur_pick(3) * cur_pick(4);
                
                if( ((overlapArea / area_true) > threshold_overlap) || ((overlapArea / area_pick) > threshold_overlap) )
                    true_bounding = [true_bounding' cur_pick']';
                    true_score = [true_score cur_scores_picked(j)];
                    break
                end
                
            end
        end
    end

end