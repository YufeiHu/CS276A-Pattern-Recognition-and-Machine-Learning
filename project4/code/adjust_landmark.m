function [landmark_adjust] = adjust_landmark(landmark)

    landmark = reshape(landmark, 1, []);
    
    landmark_pos_offset = length(landmark) / 2;
    landmark_adjust = zeros(landmark_pos_offset, 2);
    for i = 1 : landmark_pos_offset
        landmark_adjust(i, 1) = landmark(i);
        landmark_adjust(i, 2) = landmark(i + landmark_pos_offset);
    end
    

end