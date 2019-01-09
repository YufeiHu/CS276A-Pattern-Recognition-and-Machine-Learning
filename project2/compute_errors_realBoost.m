function [z_optimal, i_optimal, h_optimal] = compute_errors_realBoost(Y, features, w, B)
    
    h_max = 5;

    N_filters = size(features, 1);
    N_images = size(features, 2);
    
    z_list = ones(N_filters, 1) .* 2000;
    p_list = ones(N_filters, B) .* 2000;
    q_list = ones(N_filters, B) .* 2000;
    
%     z_optimal = N_images + 1;
%     p_optimal = zeros(B, 1);
%     q_optimal = zeros(B, 1);
%     i_optimal = 0;
    
    parfor i = 1 : N_filters
        
        feature_single = features(i, :);
        if(feature_single(1)==200)
            continue
        end
        h_pool = linspace(min(feature_single), max(feature_single), B+1);
        
        
        [p, q, z] = compute_errors_helper_realBoost(feature_single, h_pool, w, Y);
        % ==================
%         p = zeros(B, 1);
%         q = zeros(B, 1);
%         
%         for j = 1 : N_images
%             b_index = bin_index_count(feature_single(j), h_pool);
%             if(Y(j)==1)
%                 p(b_index) = p(b_index) + w(j);
%             else
%                 q(b_index) = q(b_index) + w(j);
%             end
%         end
%         
%         z = 2 * sum(sqrt(p .* q));
        % ==================
        
        z_list(i) = z;
        p_list(i, :) = p';
        q_list(i, :) = q';
        
%         if(z < z_optimal)
%             z_optimal = z;
%             i_optimal = i;
%             p_optimal = p;
%             q_optimal = q;
%         end
        
    end
    
    [z_optimal, i_optimal] = min(z_list);
    p_optimal = p_list(i_optimal, :);
    q_optimal = q_list(i_optimal, :);
    
    h_optimal = zeros(B, 1);
    for i = 1 : B
        if(q_optimal(i) == 0)
            h_optimal(i) = h_max;
        elseif(p_optimal(i) == 0)
            h_optimal(i) = -h_max;
        else
            h_optimal(i) = 0.5 * log(p_optimal(i) / q_optimal(i));
            if(h_optimal(i) > h_max)
                h_optimal(i) = h_max;
            elseif(h_optimal(i) < -h_max)
                h_optimal(i) = -h_max;
            end
        end
    end
    
    
end