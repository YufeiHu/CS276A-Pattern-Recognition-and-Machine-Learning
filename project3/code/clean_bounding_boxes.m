function [bounding_boxes_clean, scores_clean] = clean_bounding_boxes(bounding_boxes, scores, threshold_overlap)


    bounding_boxes_clean = [];
    scores_clean = [];

    while(~isempty(scores))

        bounding_boxes_cur = bounding_boxes(1, :);
        bounding_boxes_scores_cur = scores(1);
        index_delete = [1];

        for i = 2 : size(scores, 1)
            
            bounding_boxes_compare = bounding_boxes(i, :);
            bounding_boxes_scores_compare = scores(i);
            
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

        bounding_boxes_clean = [bounding_boxes_clean bounding_boxes_cur'];
        scores_clean = [scores_clean bounding_boxes_scores_cur];

        bounding_boxes(index_delete, :) = [];
        scores(index_delete) = [];

    end

    bounding_boxes_clean = bounding_boxes_clean';
    scores_clean = scores_clean';


end