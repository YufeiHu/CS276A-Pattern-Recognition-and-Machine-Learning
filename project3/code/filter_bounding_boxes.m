function [bounding_boxes_filtered, scores_filtered] = filter_bounding_boxes(bounding_boxes, scores, threshold)
    
    N = size(bounding_boxes, 1);
    delete_index = [];
    
    for i = 1 : N
        
        if(scores(i) < threshold)
            delete_index = [delete_index i];
        end
        
    end
    
    bounding_boxes(delete_index, :) = [];
    scores(delete_index, :) = [];
    
    bounding_boxes_filtered = bounding_boxes;
    scores_filtered = scores;
    
end