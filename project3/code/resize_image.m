function [I_resize] = resize_image(I, new_size)

    old_height = size(I, 1);
    old_width = size(I, 2);
    I_scale = new_size / min(old_height, old_width);
    I_resize = imresize(I, I_scale);

end