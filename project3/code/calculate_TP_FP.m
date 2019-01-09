function [cur_TP, cur_FP] = calculate_TP_FP(cur_bounding_boxes_picked, true_bbox)
    
    threshold_overlap = 0.5;
    
    if(isempty(cur_bounding_boxes_picked))
        cur_TP = 0;
        cur_FP = 0;
    elseif(isempty(true_bbox))
        cur_TP = 0;
        cur_FP = size(cur_bounding_boxes_picked, 1);
    else
        cur_TP = 0;
        cur_FP = 0;
        for i = 1 : size(cur_bounding_boxes_picked, 1)
            cur_pick = cur_bounding_boxes_picked(i, :);
            
            flag_true = 0;
            for j = 1 : size(true_bbox, 1)
                cur_true = true_bbox(j, :);
                
                overlapArea = rectint(cur_true, cur_pick);
                area_true = cur_true(3) * cur_true(4);
                area_pick = cur_pick(3) * cur_pick(4);
                
                if( ((overlapArea / area_true) > threshold_overlap) || ((overlapArea / area_pick) > threshold_overlap) )
                    flag_true = 1;
                    break
                end
                
            end
            
            if(flag_true==1)
                cur_TP = cur_TP + 1;
            else
                cur_FP = cur_FP + 1;
            end
        end
        
    end

end