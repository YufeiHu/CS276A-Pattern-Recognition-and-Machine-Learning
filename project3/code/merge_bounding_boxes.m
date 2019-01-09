function [bounding_boxes_merged, scores_merged] = merge_bounding_boxes(bounding_boxes, scores, threshold_overlap)

    
    bounding_boxes_merged = [];
    scores_merged = [];

    while(~isempty(scores))
        
        positions_cur = bounding_boxes(1, 1:2);
        width_cur = bounding_boxes(1, 3);
        height_cur = bounding_boxes(1, 4);
        
        scores_cur = scores(1);
        index_delete = [1];

        for i = 2 : size(scores, 1)

            positions_compare = bounding_boxes(i, 1:2);
            width_compare = bounding_boxes(i, 3);
            height_compare = bounding_boxes(i, 4);
            scores_compare = scores(i);
            
            overlapArea = rectint([positions_cur(1) positions_cur(2) width_cur height_cur], [positions_compare(1) positions_compare(2) width_compare height_compare]);
            area_cur = width_cur * height_cur;
            area_compare = width_compare * height_compare;
            
            if( ((overlapArea / area_cur) > threshold_overlap) || ((overlapArea / area_compare) > threshold_overlap) )
                if(scores_cur < scores_compare)
                    positions_cur = positions_compare;
                    width_cur = width_compare;
                    height_cur = height_compare;
                    scores_cur = scores_compare;
                end
                index_delete = [index_delete i];
            end

        end

        bounding_boxes_merged = [bounding_boxes_merged [positions_cur width_cur height_cur]'];
        scores_merged = [scores_merged scores_cur];

        bounding_boxes(index_delete, :) = [];
        scores(index_delete) = [];

    end

    bounding_boxes_merged = bounding_boxes_merged';
    scores_merged = scores_merged';


end