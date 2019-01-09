function [gov_labels] = compute_diff_labels(gov_votediff)

    num_pair = size(gov_votediff, 1) / 2;
    gov_labels = zeros(num_pair, num_pair * 2);
    
    for i = 1 : num_pair
        x_index = 2 * i - 1;
        result_1 = gov_votediff(x_index);
        if(result_1 > 0)
            gov_labels(i, x_index:(x_index+1)) = [1 -1];
        else
            gov_labels(i, x_index:(x_index+1)) = [-1 1];
        end
    end

end