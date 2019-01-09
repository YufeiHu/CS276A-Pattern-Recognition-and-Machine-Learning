function [face_positions_merge, face_scores_merge] = merge_boxes(face_positions, face_scores, margin_max)


    face_positions_tmp = face_positions;
    face_scores_tmp = face_scores;
    face_positions_merge = [];
    face_scores_merge = [];


    while(~isempty(face_scores_tmp))

        face_positions_cur = face_positions_tmp(1, :);
        face_scores_cur = face_scores_tmp(1);
        index_delete = [1];

        for i = 2 : size(face_scores_tmp, 1)

            face_positions_compare = face_positions_tmp(i, :);
            face_scores_compare = face_scores_tmp(i);

            if( (abs(face_positions_cur(1) - face_positions_compare(1)) < margin_max) && (abs(face_positions_cur(2) - face_positions_compare(2)) < margin_max) )
                if(face_scores_cur < face_scores_compare)
                    face_positions_cur = face_positions_compare;
                    face_scores_cur = face_scores_compare;
                end
                index_delete = [index_delete i];
            end

        end

        face_positions_merge = [face_positions_merge face_positions_cur'];
        face_scores_merge = [face_scores_merge face_scores_cur];

        face_positions_tmp(index_delete, :) = [];
        face_scores_tmp(index_delete) = [];
        
        
        
        

    end

    face_positions_merge = face_positions_merge';
    face_scores_merge = face_scores_merge';


end