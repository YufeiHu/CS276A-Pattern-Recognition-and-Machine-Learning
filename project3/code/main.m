%% =================== File header ===================
% Title: CS 276A - Pattern Recognition and Machine Learning Project 3
% Author: Yufei Hu
% Date: 11/17/2017

clc
clear
close all
set(0, 'defaultfigurecolor', [1 1 1]);

mex -setup
mex -setup C++


%% ========== Compile and load the pre-trained model ==========
disp('========== Compile and load the pre-trained model ==========');

Setup();

net = load("../../data/models/fast-rcnn-caffenet-pascal07-dagnn.mat");

net = preprocessNet(net);


%% ============= Evaluate the pre-trained model ============
disp('============= Evaluate the pre-trained model ============');
new_size = 600;


% Prepare data
I = single(imread('../../example.jpg'));
example_boxes = load('../../example_boxes.mat');
boxes = single(example_boxes.boxes);


% Resize and normalize the data
[I_resize, boxes_resize] = resize_image_boxes(I, boxes, new_size);
I_final = normalize_image(I_resize, net.meta.normalization.averageImage);


% Evaluate the model
net.eval({'data', single(I_final), 'rois', single(boxes_resize')});


% Get bounding boxes
class_index = 8;

car_delta = squeeze(net.vars(23).value);
car_delta = car_delta((4*class_index-3):4*class_index, :);
car_boxes = bbox_transform_inv(boxes, car_delta');
car_bounding_boxes = car_boxes;
car_bounding_boxes(:, 3) = car_boxes(:, 3) - car_boxes(:, 1);
car_bounding_boxes(:, 4) = car_boxes(:, 4) - car_boxes(:, 2);

car_scores = squeeze(net.vars(24).value)';
car_scores = car_scores(:, class_index);


% Adjust bounding boxes so they don't go out of range
image_width = size(I_final, 2);
image_height = size(I_final, 1);
[car_bounding_boxes_adjust, car_scores_adjust] = adjust_bounding_boxes(car_bounding_boxes, car_scores, image_width, image_height);


% Filter bounding boxes whose scores are below a threshold
threshold_filter = 0;
[car_bounding_boxes_filtered, car_scores_filtered] = filter_bounding_boxes(car_bounding_boxes_adjust, car_scores_adjust, threshold_filter);


% Merge overlapping bounding boxes
threshold_overlap = 0.7;
[car_bounding_boxes_merged, car_scores_merged] = merge_bounding_boxes(car_bounding_boxes_filtered, car_scores_filtered, threshold_overlap);


% Clean overlapping bounding boxes
threshold_clean = 0.7;
[car_bounding_boxes_clean, car_scores_clean] = clean_bounding_boxes(car_bounding_boxes_merged, car_scores_merged, threshold_clean);


% Pick top 100 bounding boxes
[car_bounding_boxes_picked, car_scores_picked] = pick_bounding_boxes(car_bounding_boxes_clean, car_scores_clean);


% Find optimal threshold
threshold_test = linspace(0, 1, 100);

num_detected = zeros(size(threshold_test));
for i = 1 : size(threshold_test, 2)
    num_detected(i) = size(find(car_scores_picked>threshold_test(i)), 1);
end

figure(1)
plot(threshold_test, num_detected, 'LineWidth', 3);
title('Number of Detections V.S. Thresholds', 'FontSize', 20);
xlabel('Thresholds', 'FontSize', 20);
ylabel('Number of Detections', 'FontSize', 20);
grid on


% Plot bounding boxes
threshold_chosen = 0.6;
figure(2);
hold on
imshow(imread('../../example.jpg'));
for i = 1 : size(car_scores_picked, 1)
    if(car_scores_picked(i) > threshold_chosen)
        rectangle('Position', car_bounding_boxes_picked(i, :), 'EdgeColor', 'r', 'LineWidth', 2);
        str_score = sprintf('%0.2f', car_scores_picked(i));
        text(double(car_bounding_boxes_picked(i, 1)+1), double(car_bounding_boxes_picked(i, 2)-12), str_score, ...
            'HorizontalAlignment', 'left', 'FontSize', 10, 'Color', 'white', 'FontWeight', 'bold', 'BackgroundColor', 'red', ...
            'VerticalAlignment', 'top');
    end
end
hold off


%% ========= Object detection on Pascal VOC 2007 dataset ========
disp('========= Object detection on Pascal VOC 2007 dataset ========');
new_size = 600;
threshold_chosen = 0.1;
image_interest = 679; % 13, 30, 73, 679


% Prepare data
pascal = load('../../data/SSW/SelectiveSearchVOC2007test.mat');
pascal_boxes = pascal.boxes;
pascal_image_index = pascal.images;
pascal_image_num = size(pascal_image_index, 1);
class_dic = net.meta.classes.name;


TP = zeros(20, 1);
FP = zeros(20, 1);
car_PR = [];
image_interest_bounding = cell(20, 2);


% Analyze each image
for i = 1 : pascal_image_num %1 : pascal_image_num
    
    tic;
    % Read image and boxes
    I_path = strcat( '../../data/images/', pascal_image_index{i}, '.jpg' );
    I = single(imread(I_path));
    boxes = single(pascal_boxes{i});
    
    
    % Read annotations
    anno_path = strcat( '../../data/annotations/', pascal_image_index{i}, '.xml' );
    anno = PASreadrecord(anno_path);
    anno_num = size(anno.objects, 2);
    anno_class = zeros(anno_num, 1);
    anno_boxes = zeros(anno_num, 4);
    for j = 1 : anno_num
        anno_class(j) = findClass(class_dic, anno.objects(j).class);
        anno_boxes(j, :) = anno.objects(j).bbox;
    end
    

    % Resize and normalize the data
    [I_resize, boxes_resize] = resize_image_boxes(I, boxes, new_size);
    I_final = normalize_image(I_resize, net.meta.normalization.averageImage);
    
    
    % Evaluate the model
    net.eval({'data', single(I_final), 'rois', single(boxes_resize')});
    
    
    all_delta = squeeze(net.vars(23).value);
    all_scores = squeeze(net.vars(24).value)';
    image_width = size(I, 2);
    image_height = size(I, 1);
    cur_PR = [];
    for class_index = 2 : 21 %2 : 21
        
        % Get bounding boxes
        cur_delta = all_delta((4*class_index-3) : 4*class_index, :);
        cur_boxes = bbox_transform_inv(boxes, cur_delta');
        cur_bounding_boxes = cur_boxes;
        cur_bounding_boxes(:, 3) = cur_boxes(:, 3) - cur_boxes(:, 1);
        cur_bounding_boxes(:, 4) = cur_boxes(:, 4) - cur_boxes(:, 2);
        
        cur_scores = all_scores(:, class_index);
        
        
        % Adjust bounding boxes so they don't go out of range
        [cur_bounding_boxes_adjust, cur_scores_adjust] = adjust_bounding_boxes(cur_bounding_boxes, cur_scores, image_width, image_height);
        
        
        % Filter bounding boxes whose scores are below a threshold
        threshold_filter = threshold_chosen;
        [cur_bounding_boxes_filtered, cur_scores_filtered] = filter_bounding_boxes(cur_bounding_boxes_adjust, cur_scores_adjust, threshold_filter);
        if(class_index==8)
            car_threshold_filter = 0.01;
            [car_bounding_boxes_filtered, car_scores_filtered] = filter_bounding_boxes(cur_bounding_boxes_adjust, cur_scores_adjust, car_threshold_filter);
        end
        
        
        % Merge overlapping bounding boxes
        threshold_overlap = 0.7;
        [cur_bounding_boxes_merged, cur_scores_merged] = merge_bounding_boxes(cur_bounding_boxes_filtered, cur_scores_filtered, threshold_overlap);
        if(class_index==8)
            [car_bounding_boxes_merged, car_scores_merged] = merge_bounding_boxes(car_bounding_boxes_filtered, car_scores_filtered, threshold_overlap);
        end
        
        
        % Clean overlapping bounding boxes
        threshold_clean = 0.7;
        [cur_bounding_boxes_clean, cur_scores_clean] = clean_bounding_boxes(cur_bounding_boxes_merged, cur_scores_merged, threshold_clean);
        if(class_index==8)
            [car_bounding_boxes_clean, car_scores_clean] = clean_bounding_boxes(car_bounding_boxes_merged, car_scores_merged, threshold_clean);
        end
        
        
        % Pick top 100 bounding boxes
        [cur_bounding_boxes_picked, cur_scores_picked] = pick_bounding_boxes(cur_bounding_boxes_clean, cur_scores_clean);
        if(class_index==8)
            [car_bounding_boxes_picked, car_scores_picked] = pick_bounding_boxes(car_bounding_boxes_clean, car_scores_clean);
        end
        
        
        anno_index = find(anno_class==class_index);
        true_boxes = anno_boxes(anno_index, :);
        true_bbox = true_boxes;
        true_bbox(:, 3) = true_boxes(:, 3) - true_boxes(:, 1);
        true_bbox(:, 4) = true_boxes(:, 4) - true_boxes(:, 2);
        
        
        [cur_TP, cur_FP] = calculate_TP_FP(cur_bounding_boxes_picked, true_bbox);
        TP(class_index-1) = TP(class_index-1) + cur_TP;
        FP(class_index-1) = FP(class_index-1) + cur_FP;
        
        
        % VIP access for the car class
        if(class_index==8)
            cur_PR = calculate_car_PR(car_bounding_boxes_picked, car_scores_picked, true_bbox);
            car_PR = [car_PR' cur_PR']';
        end
        
        if(i==image_interest)
            [true_bounding, true_score] = calculate_bounding_box(cur_bounding_boxes_picked, cur_scores_picked, true_bbox);
            image_interest_bounding{class_index-1, 1} = true_bounding;
            image_interest_bounding{class_index-1, 2} = true_score;
            
%             image_interest_bounding{class_index-1, 1} = cur_bounding_boxes_picked;
%             image_interest_bounding{class_index-1, 2} = cur_scores_picked;
        end
        
    end
    
    
    if(i==image_interest)
        
        colorspec = {[0.9 0.3 0.2]; [0.1 0.2 0.8]; [0.6 0.6 0.4]; [0.2 0.1 0.3]; [0.32 0.6 0.42]; ...
                           [0.32 0.6 0.42]; [0.32 0.6 0.42]; [0.32 0.6 0.42]; [0.92 0.6 0.42]; [0.32 0.6 0.42]; ...
                           [0.91 0.75 0.32]; [0.78 0.56 0.42]; [0.54 0.91 0.3]; [0.54 0.52 0.74]; [0.14 0.64 0.64]; ...
                           [0.37 0.13 0.56]; [0.11 0.69 0.46]; [0.72 0.77 0.37]; [0.32 0.94 0.12]; [0.12 0.86 0.97]};
        figure(3);
        hold on
        imshow(imread(I_path));
        for ii = 1 : 20
            interest_bbox = image_interest_bounding{ii, 1};
            interest_score = image_interest_bounding{ii, 2};
            if(~isempty(interest_score))
                for jj = 1 : length(interest_score)
                    if(interest_score(jj) > threshold_chosen)
                        rectangle('Position', interest_bbox(jj, :), 'EdgeColor', colorspec{ii}, 'LineWidth', 2);
                        str_score = sprintf('%s: %0.2f', class_dic{ii+1}, interest_score(jj));
                        text(double(interest_bbox(jj, 1)+1), double(interest_bbox(jj, 2)-10), str_score, ...
                            'HorizontalAlignment', 'left', 'FontSize', 10, 'Color', 'black', 'FontWeight', 'bold', 'BackgroundColor', colorspec{ii}, ...
                            'VerticalAlignment', 'top');
                    end
                end
            end
        end
        hold off
        
    end
    
    fprintf('Detecting the %dth image took %.2f secs.\n', i, toc);
    
end


%% ================== Result analysis =================
disp('================== Result analysis =================');
% Calculate precisions
total_p = TP + FP;
AP = TP ./ total_p;
mAP = sum(TP) / (sum(TP) + sum(FP));


% Draw PR curve for car class
threshold_PR = linspace(0, 1, 1000);
precision = zeros(size(threshold_PR));
recall = zeros(size(threshold_PR));

for i = 1 : length(threshold_PR)
    fprintf('Processing the %dth threshold\n', i);
    PR_tmp = zeros(size(car_PR, 1));
    PR_tmp(car_PR(:, 2)>threshold_PR(i)) = 1;
    TP_tmp = 0;
    FP_tmp = 0;
    FN_tmp = 0;
    
    for j = 1 : length(PR_tmp)
        if((car_PR(j, 1)==1) && (PR_tmp(j)==1))
            TP_tmp = TP_tmp + 1;
        elseif((car_PR(j, 1)==1) && (PR_tmp(j)==0))
            FN_tmp = FN_tmp + 1;
        elseif((car_PR(j, 1)==0) && (PR_tmp(j)==1))
            FP_tmp = FP_tmp + 1;
        end
    end
    
    precision(i) = TP_tmp / (TP_tmp + FP_tmp);
    recall(i) = TP_tmp / (TP_tmp + FN_tmp);
end

figure(4)
plot(recall, precision, 'LineWidth', 3);
str_title = sprintf('Precision-recall curve for car class, mAP: %.4f', mAP);
title(str_title);
xlabel('recall');
ylabel('precision');
grid on
