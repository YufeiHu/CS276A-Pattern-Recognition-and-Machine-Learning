function [bin_index] = bin_index_count(data, bins)

    B = length(bins) - 1;
    
    for i = 1 : B
        if ((data>bins(i)) && (data<bins(i+1)))
            break;
        end
    end
    
    bin_index = i;

end