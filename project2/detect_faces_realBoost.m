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


% load trained strong classifier
load('training_results_realBoost.mat');


% load filter_H
T = 200;
filter_H = cell(T, 2);
for i = 1 : T
    filter_H{i, 1} = filters{index_H(i), 1};
    filter_H{i, 2} = filters{index_H(i), 2};
end



%% ========= Detect faces using testing images ==========
% User definition
I_test_1_rgb = imread(sprintf('Testing_Images/Face_1.jpg'));
I_test_1 = rgb2gray(imread(sprintf('Testing_Images/Face_1.jpg')));
I_scale = [1 0.8 0.6 0.4 0.35 0.3 0.25 0.2 0.15 0.1];
margin_max = 5;
threshold = 0;
B = 30;


% Use Strong Classifier to detect
face_positions_merge = cell(1, length(I_scale));
face_scores_merge = cell(1, length(I_scale));

for i = 1 : length(I_scale)
    I_test = imresize(I_test_1, I_scale(i));
    tic;
    [face_positions, face_scores] = detect_face_realBoost(I_test, filter_H, h_H, threshold, B);
    [face_positions_merge{i}, face_scores_merge{i}] = merge_boxes(face_positions, face_scores, margin_max * I_scale(i));
    fprintf('Detecting faces with scale %.1f took %.2f secs.\n', I_scale(i), toc);
end


% Merge close boxes
face_positions_merge_unified = cell(1, length(I_scale));
for i = 1 : length(I_scale)
    face_positions_merge_unified{i} = ((face_positions_merge{i} - 1) ./ I_scale(i)) + 1;
end


% Generate bounding boxes
[bounding_boxes, bounding_boxes_scores] = merge_boxes_unified(face_positions_merge_unified, face_scores_merge, I_scale);
[bounding_boxes, bounding_boxes_scores] = clean_bounding_boxes(bounding_boxes, bounding_boxes_scores);
% save('I_test_1_HN.mat', 'face_positions_merge', 'face_scores_merge', 'face_positions_merge_unified', 'bounding_boxes', 'bounding_boxes_scores');



%% ============= Plotting region =============
% Plot merged faces
% threshold_1 = 1.2;
% threshold_2 = 2;
% figure(1);
% hold on
% imshow(imresize(I_test_1, I_scale(1)));
% for k = 1 : length(I_scale)
%     for i = 1 : size(face_positions_merge_unified{k}, 1)
%         if(face_scores_merge{k}(i) < threshold_1)
% %             rectangle('Position', [face_positions_merge_unified{k}(i, 1), face_positions_merge_unified{k}(i, 2), 16/I_scale(k), 16/I_scale(k)], 'EdgeColor', 'r');
%         elseif((face_scores_merge{k}(i) > threshold_1) && (face_scores_merge{k}(i) < threshold_2))
%             rectangle('Position', [face_positions_merge_unified{k}(i, 1), face_positions_merge_unified{k}(i, 2), 16/I_scale(k), 16/I_scale(k)], 'EdgeColor', 'y');
%         else
%             rectangle('Position', [face_positions_merge_unified{k}(i, 1), face_positions_merge_unified{k}(i, 2), 16/I_scale(k), 16/I_scale(k)], 'EdgeColor', 'g');
%         end
%     end
% end
% hold off


% Plot bounding boxes
threshold_0 = 1.7;
threshold_1 = 1.8;
threshold_2 = 2.4;
figure(2);
hold on
imshow(imresize(I_test_1_rgb, I_scale(1)));
for i = 1 : size(bounding_boxes_scores, 1)
    if((bounding_boxes_scores(i) > threshold_0) && (bounding_boxes_scores(i) < threshold_1))
        rectangle('Position', bounding_boxes(i, :), 'EdgeColor', 'r', 'LineWidth', 2);
    elseif((bounding_boxes_scores(i) > threshold_1) && (bounding_boxes_scores(i) < threshold_2))
        rectangle('Position', bounding_boxes(i, :), 'EdgeColor', 'y', 'LineWidth', 2);
    elseif(bounding_boxes_scores(i) > threshold_2)
        rectangle('Position', bounding_boxes(i, :), 'EdgeColor', 'g', 'LineWidth', 2);
    end
end
hold off



%% ============= Helper functions =============
function [face_positions, face_scores] = detect_face_realBoost(I, filter_H, h_H, threshold, B)

    image_height = size(I, 1);
    image_width = size(I, 2);
    
    stride = 1;
    window_size = 16;
    
    num_horizon = floor((image_width - window_size) / stride) + 1;
    num_vertical = floor((image_height - window_size) / stride) + 1;
    
    i_max = num_vertical * stride;
    j_max = num_horizon * stride;
    
    x_pos = [];
    y_pos = [];
    
    face_scores = [];
    
    parfor i = 1: i_max
        
        for j = 1: j_max
            
            I_patch = zeros(1, 16, 16);
            I_patch(1, :, :) = I(i : (i + window_size - 1), j : (j + window_size - 1));
            I_patch = normalize(I_patch);
            II_patch = integral(I_patch);
            features = compute_features(II_patch, filter_H);
            
            N_filters = size(features, 1);
            N_images = size(features, 2);
            
            
            responses = zeros(N_filters, N_images);
            for m = 1 : N_filters
                feature_single = features(m, :);
                h_pool = linspace(min(feature_single), max(feature_single), B+1);
                
                responses_single = zeros(1, N_images);
                for ii = 1 : N_images
                    b_index = bin_index_count(feature_single(ii), h_pool);
                    responses_single(ii) = h_H(m, b_index);
                end
                responses(m, :) = responses_single;
            end
            

            scores = sum(responses);
            
            if(scores>threshold)
                
                face_scores = [face_scores scores];
                x_pos = [x_pos j];
                y_pos = [y_pos i];
                
            end
            
        end
        
    end
    
    face_positions = [x_pos' y_pos'];
    face_scores = face_scores';

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