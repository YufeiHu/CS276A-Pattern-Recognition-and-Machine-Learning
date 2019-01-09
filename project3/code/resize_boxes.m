function [boxes_resize] = resize_boxes(I, boxes, new_size)

    old_height = size(I, 1);
    old_width = size(I, 2);
    I_scale = new_size / min(old_height, old_width);
    
    boxes_resize = zeros(size(boxes, 1), size(boxes, 2));
    boxes_resize(:, :) = boxes(:, :) .* I_scale + 1;

end