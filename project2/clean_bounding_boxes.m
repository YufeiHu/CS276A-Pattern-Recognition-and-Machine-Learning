function [bounding_boxes, bounding_boxes_scores] = clean_bounding_boxes(bounding_boxes_input, bounding_boxes_scores_input)

    window_size = 16;
    threshold_overlap = 0.1;

    bounding_boxes = [];
    bounding_boxes_scores = [];

    while(~isempty(bounding_boxes_scores_input))

        bounding_boxes_cur = bounding_boxes_input(1, :);
        bounding_boxes_scores_cur = bounding_boxes_scores_input(1);
        index_delete = [1];

        for i = 2 : size(bounding_boxes_scores_input, 1)
            
            bounding_boxes_compare = bounding_boxes_input(i, :);
            bounding_boxes_scores_compare = bounding_boxes_scores_input(i);
            
            overlapArea = rectint(bounding_boxes_compare, bounding_boxes_cur);
            area_cur = bounding_boxes_cur(3) * bounding_boxes_cur(4);
            area_compare = bounding_boxes_compare(3) * bounding_boxes_compare(4);
            
            if( ((overlapArea / area_cur) > threshold_overlap) || ((overlapArea / area_compare) > threshold_overlap) )
                if(bounding_boxes_scores_cur < bounding_boxes_scores_compare)
                    bounding_boxes_cur = bounding_boxes_compare;
                    bounding_boxes_scores_cur = bounding_boxes_scores_compare;
                end
                index_delete = [index_delete i];
            end

        end

        bounding_boxes = [bounding_boxes bounding_boxes_cur'];
        bounding_boxes_scores = [bounding_boxes_scores bounding_boxes_scores_cur];

        bounding_boxes_input(index_delete, :) = [];
        bounding_boxes_scores_input(index_delete) = [];

    end

    bounding_boxes = bounding_boxes';
    bounding_boxes_scores = bounding_boxes_scores';


end