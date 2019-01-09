function [bounding_boxes, bounding_boxes_scores] = merge_boxes_unified(face_positions_merge_unified, face_scores_merge, I_scale)

    window_size = 16;
    threshold_overlap = 0.1;
    num_scales = length(I_scale);
    face_positions_tmp = [];
    face_scores_tmp = [];
    
    for i = 1 : num_scales
        
        num_boxes = size(face_positions_merge_unified{i}, 1);
        tmp = [face_positions_merge_unified{i} ones(num_boxes, 1) .* (window_size / I_scale(i))];
        face_positions_tmp = [face_positions_tmp  tmp'];
        face_scores_tmp = [face_scores_tmp face_scores_merge{i}'];
    end
    
    face_positions_tmp = face_positions_tmp';
    face_scores_tmp = face_scores_tmp';

    
    bounding_boxes = [];
    bounding_boxes_scores = [];

    while(~isempty(face_scores_tmp))

        face_positions_cur = face_positions_tmp(1, 1:2);
        face_width_cur = face_positions_tmp(1, 3);
        face_scores_cur = face_scores_tmp(1);
        index_delete = [1];

        for i = 2 : size(face_scores_tmp, 1)

            face_positions_compare = face_positions_tmp(i, 1:2);
            face_width_compare = face_positions_tmp(i, 3);
            face_scores_compare = face_scores_tmp(i);
            
            overlapArea = rectint([face_positions_cur(2) face_positions_cur(1) face_width_cur face_width_cur], [face_positions_compare(2) face_positions_compare(1) face_width_compare face_width_compare]);
            area_cur = face_width_cur * face_width_cur;
            area_compare = face_width_compare * face_width_compare;
            
            if( ((overlapArea / area_cur) > threshold_overlap) || ((overlapArea / area_compare) > threshold_overlap) )
                if(face_scores_cur < face_scores_compare)
                    face_positions_cur = face_positions_compare;
                    face_width_cur = face_width_compare;
                    face_scores_cur = face_scores_compare;
                end
                index_delete = [index_delete i];
            end

        end

        bounding_boxes = [bounding_boxes [face_positions_cur face_width_cur face_width_cur]'];
        bounding_boxes_scores = [bounding_boxes_scores face_scores_cur];

        face_positions_tmp(index_delete, :) = [];
        face_scores_tmp(index_delete) = [];

    end

    bounding_boxes = bounding_boxes';
    bounding_boxes_scores = bounding_boxes_scores';


end