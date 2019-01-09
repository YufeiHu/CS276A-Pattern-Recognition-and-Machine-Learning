clc
clear
close all


%% ============= Preparation =============
flag_parpool = 1;


% parpool
if flag_parpool
    delete(gcp('nocreate'));
    parpool(4);
end


% construct filters
A = filters_A();
B = filters_B();
C = filters_C();
D = filters_D();
filters = [A; B; C; D];


% load trained strong classifier
load('training_results.mat');


% load filter_H
T = 200;
filter_H = cell(T, 2);
for i = 1 : T
    filter_H{i, 1} = filters{index_H(i), 1};
    filter_H{i, 2} = filters{index_H(i), 2};
end



%% ============= Hard-negatives mining ==============
% User definition
num_HN = 3;
I_scale = [1 0.8 0.6 0.4 0.35 0.3 0.25 0.2 0.15 0.1];
threshold = 0;


% Hard-negatives mining
HN = zeros(1, 16, 16);
m = 1;

for i = 1 : num_HN
    I_test = rgb2gray(imread(sprintf('Testing_Images/Non_face_%d.jpg', i)));
    for k = 1 : length(I_scale)
        I_test_rescale = imresize(I_test, I_scale(k));
        tic;
        [HN_tmp] = detect_HN(I_test_rescale, filter_H, param_H, alpha_H, threshold);
        num_patches = size(HN_tmp, 2) / 16;
        for j = 1 : num_patches
            HN(m, :, :) = HN_tmp(1:16, 1 + 16 * (j - 1) : 16 * j);
            m = m + 1;
        end
        fprintf('Hard-negative mining the %dth image with scale %.1f took %.2f secs.\n', i, I_scale(k), toc);
    end
end


%% ============= Helper functions =============
function [HN_tmp] = detect_HN(I, filter_H, param_H, alpha_H, threshold)

    image_height = size(I, 1);
    image_width = size(I, 2);
    
    stride = 1;
    window_size = 16;
    
    num_horizon = floor((image_width - window_size) / stride) + 1;
    num_vertical = floor((image_height - window_size) / stride) + 1;
    
    i_max = num_vertical * stride;
    j_max = num_horizon * stride;
    
    HN_tmp = [];
    
    parfor i = 1: i_max
        for j = 1: j_max
            
            I_patch = zeros(1, 16, 16);
            I_patch(1, :, :) = I(i : (i + window_size - 1), j : (j + window_size - 1));
            I_patch_nor = normalize(I_patch);
            II_patch = integral(I_patch_nor);
            features = compute_features(II_patch, filter_H);
            
            N_filters = size(features, 1);
            N_images = size(features, 2);
            
            responses = ones(N_filters, N_images) .* param_H(:, 2);
            tmp = -param_H(:, 2);
            responses(features<param_H(:, 1)) = tmp(features<param_H(:, 1));
            
            scores = sum(responses .* alpha_H);
            
            if(scores>threshold)
                HN_tmp = [HN_tmp I_patch];
            end
            
        end
    end
    HN_tmp = squeeze(HN_tmp);
    HN_tmp = HN_tmp';
end


function [scores, Y_hat] = strong_classifier_verify(alpha_H, param_H, index_H, features, N_weak)

    N_filters = size(features, 1);
    N_images = size(features, 2);
    
    scores = zeros(N_images, 1);
    Y_hat = ones(N_images, 1);
    
    for i = 1:N_weak
        
        responses = compute_response(param_H(i, :), index_H(i), features);
        scores = scores + alpha_H(i) .* responses;
        
    end
    
    Y_hat(scores<0) = -1;

end


function features = compute_features(II, filters)
    features = zeros(size(filters, 1), size(II, 1));
    for j = 1:size(filters, 1)
        [rects1, rects2] = filters{j,:};
        features(j,:) = apply_filter(II, rects1, rects2);
    end
end


function I = normalize(I)
    [N,~,~] = size(I);
    for i = 1:N
        image = I(i,:,:);
        sigma = std(image(:));
        I(i,:,:) = I(i,:,:) / sigma;
    end
end


function II = integral(I)
    [N,H,W] = size(I);
    II = zeros(N,H+1,W+1);
    for i = 1:N
        image = squeeze(I(i,:,:));
        II(i,2:H+1,2:W+1) = cumsum(cumsum(double(image), 1), 2);
    end
end


function sum = apply_filter(II, rects1, rects2)
    sum = 0;
    
    % white rects
    for k = 1:size(rects1,1)
        r1 = rects1(k,:);
        w = r1(3);
        h = r1(4);
        sum = sum + sum_rect(II, [0, 0], r1) / (w * h * 255);
    end
    
    % black rects
    for k = 1:size(rects2,1)
        r2 = rects2(k,:);
        w = r2(3);
        h = r2(4);
        sum = sum - sum_rect(II, [0, 0], r2) / (w * h * 255);
    end
    
end


function result = sum_rect(II, offset, rect)
    x_off = offset(1);
    y_off = offset(2);

    x = rect(1);
    y = rect(2);
    w = rect(3);
    h = rect(4);

    a1 = II(:, y_off + y + h, x_off + x + w);
    a2 = II(:, y_off + y + h, x_off + x);
    a3 = II(:, y_off + y,     x_off + x + w);
    a4 = II(:, y_off + y,     x_off + x);

    result = a1 - a2 - a3 + a4;
end


function rects = filters_A()
    count = 1;
    w_min = 4;
    h_min = 4;
    w_max = 16;
    h_max = 16;
    rects = cell(1,2);
    for w = w_min:2:w_max
        for h = h_min:h_max
            for x = 1:(w_max-w)
                for y = 1:(h_max-h)
                    r1_x = x;
                    r1_y = y;
                    r1_w = w/2;
                    r1_h = h;
                    r1 = [r1_x, r1_y, r1_w, r1_h];

                    r2_x = r1_x + r1_w;
                    r2_y = r1_y;
                    r2_w = w/2;
                    r2_h = h;
                    r2 = [r2_x, r2_y, r2_w, r2_h];

                    rects{count, 1} = r1; % white
                    rects{count, 2} = r2; % black
                    count = count + 1;
                end
            end
        end
    end
end


function rects = filters_B()
    count = 1;
    w_min = 4;
    h_min = 4;
    w_max = 16;
    h_max = 16;
    rects = cell(1,2);
    for w = w_min:w_max
        for h = h_min:2:h_max
            for x = 1:(w_max-w)
                for y = 1:(h_max-h)
                    r1_x = x;
                    r1_y = y;
                    r1_w = w;
                    r1_h = h/2;
                    r1 = [r1_x, r1_y, r1_w, r1_h];

                    r2_x = r1_x;
                    r2_y = r1_y + r1_h;
                    r2_w = w;
                    r2_h = h/2;
                    r2 = [r2_x, r2_y, r2_w, r2_h];

                    rects{count, 1} = r2; % white
                    rects{count, 2} = r1; % black
                    count = count + 1;
                end
            end
        end
    end
end


function rects = filters_C()
    count = 1;
    w_min = 6;
    h_min = 4;
    w_max = 16;
    h_max = 16;
    rects = cell(1,2);
    for w = w_min:3:w_max
        for h = h_min:h_max
            for x = 1:(w_max-w)
                for y = 1:(h_max-h)
                    r1_x = x;
                    r1_y = y;
                    r1_w = w/3;
                    r1_h = h;
                    r1 = [r1_x, r1_y, r1_w, r1_h];

                    r2_x = r1_x + r1_w;
                    r2_y = r1_y;
                    r2_w = w/3;
                    r2_h = h;
                    r2 = [r2_x, r2_y, r2_w, r2_h];

                    r3_x = r1_x + r1_w + r2_w;
                    r3_y = r1_y;
                    r3_w = w/3;
                    r3_h = h;
                    r3 = [r3_x, r3_y, r3_w, r3_h];

                    rects{count, 1} = [r1; r3]; % white
                    rects{count, 2} = r2; % black
                    count = count + 1;
                end
            end
        end
    end
end


function rects = filters_D()
    count = 1;
    w_min = 6;
    h_min = 6;
    w_max = 16;
    h_max = 16;
    rects = cell(1,2);
    for w = w_min:2:w_max
        for h = h_min:2:h_max
            for x = 1:(w_max-w)
                for y = 1:(h_max-h)
                    r1_x = x;
                    r1_y = y;
                    r1_w = w/2;
                    r1_h = h/2;
                    r1 = [r1_x, r1_y, r1_w, r1_h];

                    r2_x = r1_x+r1_w;
                    r2_y = r1_y;
                    r2_w = w/2;
                    r2_h = h/2;
                    r2 = [r2_x, r2_y, r2_w, r2_h];

                    r3_x = x;
                    r3_y = r1_y+r1_h;
                    r3_w = w/2;
                    r3_h = h/2;
                    r3 = [r3_x, r3_y, r3_w, r3_h];

                    r4_x = r1_x+r1_w;
                    r4_y = r1_y+r2_h;
                    r4_w = w/2;
                    r4_h = h/2;
                    r4 = [r4_x, r4_y, r4_w, r4_h];

                    rects{count, 1} = [r2; r3]; % white
                    rects{count, 2} = [r1; r4]; % black
                    count = count + 1;
                end
            end
        end
    end
end