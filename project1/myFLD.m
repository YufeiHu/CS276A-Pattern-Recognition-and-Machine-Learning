%% =================== File header ===================
% Title: CS 276A - Pattern Recognition and Machine Learning Project 1
% Subtitle: Helper function for calculating w for FLD
% Author: Yufei Hu
% Date: 10/21/2017


function w = myFLD(female_train_norm, male_train_norm, U, num_eigenfaces)


    female_b = ( female_train_norm' * U(:, 1:num_eigenfaces) )';
    male_b = ( male_train_norm' * U(:, 1:num_eigenfaces) )';
    
    
    female_mean_b = mean(female_b, 2);
    male_mean_b = mean(male_b, 2);
    
    
    female_b_norm = bsxfun(@minus, female_b, female_mean_b);
    male_b_norm = bsxfun(@minus, male_b, male_mean_b);
    
    
    Sw = male_b_norm * male_b_norm' + female_b_norm * female_b_norm';
    w_b = Sw \ (female_mean_b - male_mean_b);
    
    
    w = U(:, 1:num_eigenfaces) * w_b;

    
end