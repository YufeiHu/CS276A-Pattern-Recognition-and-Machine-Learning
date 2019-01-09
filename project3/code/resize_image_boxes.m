function [I_resize, boxes_resize] = resize_image_boxes(I, boxes, new_size)


    old_height = size(I, 1);
    old_width = size(I, 2);
    I_scale = new_size / min(old_height, old_width);
    
    
    I_resize = imresize(I, I_scale);
    
    
    boxes_resize = zeros(size(boxes, 1), size(boxes, 2) + 1);
    boxes_resize(:, 1) = 1;
    boxes_resize(:, 2:5) = boxes(:, :) .* I_scale + 1;

    
end