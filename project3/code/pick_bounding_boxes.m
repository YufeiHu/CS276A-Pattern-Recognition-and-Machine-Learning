function [bounding_boxes_picked, scores_picked] = pick_bounding_boxes(bounding_boxes, scores)
    
    num_max = 100;
    
    [~, top_index] = sort(scores, 'descend');
    
    if(size(top_index, 1)<num_max)
        bounding_boxes_picked = bounding_boxes;
        scores_picked = scores;
    else
        bounding_boxes_picked = bounding_boxes(top_index(1:num_max), :);
        scores_picked = scores(top_index(1:num_max));
    end
    
end