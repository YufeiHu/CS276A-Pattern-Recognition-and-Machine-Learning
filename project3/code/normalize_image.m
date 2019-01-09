function [I_norm] = normalize_image(I, I_average)

    I_norm = bsxfun(@minus, I, I_average);
    
end