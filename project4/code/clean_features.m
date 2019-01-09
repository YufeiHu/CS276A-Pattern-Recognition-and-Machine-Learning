function [features_clean, index_del] = clean_features(features)


    features_clean = [];
    index_del = [];
    ncol_ori = size(features, 2);
    
    
    for i = 1 : ncol_ori
        
        ncol_cur = size(features_clean, 2);
        flag_append = 1;
        
        for j = 1 : ncol_cur
            if(isequal(features(:, i), features_clean(:, j)))
                flag_append = 0;
                index_del = [index_del i];
                break
            end
        end
        
        if(flag_append==1)
            features_clean = [features_clean features(:, i)];
        end
            
    end
    

end