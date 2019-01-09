function [im_warped] = warp_image(im, landmark_adjust, ML_pos)

    im_warped = zeros(size(im));
    for i = 1 : size(im, 3)
        im_warped(:, :, i) = uint8(warpImage_new(im(:, :, i), landmark_adjust, ML_pos));
    end
    
end