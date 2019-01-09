clc
clear
close all


%% ============= Preparation =============
flag_data_subset = 0;
flag_extract_features = 0;
flag_parpool = 1;
flag_hardnegative = 1;


% parpool
if flag_parpool
    delete(gcp('nocreate'));
    parpool(4);
end


% hard negatives
if flag_hardnegative
    load('hard_negatives.mat');
    N_HN = size(HN, 1);
end


% constants
if flag_data_subset
    N_pos = 100;
    N_neg = 100;
else
    N_pos = 11838; %11838
    N_neg = 35356; %45356
end
N = N_pos + N_neg;
w = 16;
h = 16;


% load images
if flag_hardnegative
    if flag_extract_features
        tic;
        I = zeros(N+N_HN, h, w);
        Y = zeros(N+N_HN, 1);
        for i=1:N_pos
            I(i,:,:) = rgb2gray(imread(sprintf('newface16/face16_%06d.bmp',i), 'bmp'));
            Y(i) = 1;
        end
        for i=1:N_neg
            I(N_pos+i,:,:) = rgb2gray(imread(sprintf('nonface16/nonface16_%06d.bmp',i), 'bmp'));
            Y(N_pos+i) = -1;
        end
        for i=1:N_HN
            I(N+i,:,:) = HN(i, :, :);
            Y(N+i) = -1;
        end
        fprintf('Loading images and hard negatives took %.2f secs.\n', toc);
    end
else
    if flag_extract_features
        tic;
        I = zeros(N, h, w);
        Y = zeros(N, 1);
        for i=1:N_pos
            I(i,:,:) = rgb2gray(imread(sprintf('newface16/face16_%06d.bmp',i), 'bmp'));
            Y(i) = 1;
        end
        for i=1:N_neg
            I(N_pos+i,:,:) = rgb2gray(imread(sprintf('nonface16/nonface16_%06d.bmp',i), 'bmp'));
            Y(N_pos+i) = -1;
        end
        fprintf('Loading images took %.2f secs.\n', toc);
    end
end



% construct filters
A = filters_A();
B = filters_B();
C = filters_C();
D = filters_D();
if flag_data_subset
    filters = [A(1:250,:); B(1:250,:); C(1:250,:); D(1:250,:)];
else
    filters = [A; B; C; D];
end


% extract features
if flag_extract_features
    tic;
    I = normalize(I);
    II = integral(I);
    features = compute_features(II, filters);
    if flag_hardnegative
        save('features_HN.mat', '-v7.3', 'features','Y');
        fprintf('Extracting %d features from %d images and hard negatives took %.2f secs.\n', size(filters, 1), N+N_HN, toc);
    else
        save('features.mat', '-v7.3', 'features','Y');
        fprintf('Extracting %d features from %d images took %.2f secs.\n', size(filters, 1), N, toc);
    end
else
    if flag_hardnegative
        load('features_HN.mat','features','Y');
    else
        load('features.mat','features','Y');
    end
end


%% ============= Boosting =============
if flag_hardnegative
    N = N + N_HN;
end
N_filters = size(features, 1);
N_images = size(features, 2);
T = 200;


w = ones(N, 1) ./ N;
alpha_H = zeros(T, 1);
param_H = zeros(T, 2);
index_H = zeros(T, 1);
error_weak = zeros(1000, 5);
training_error = zeros(T, 1);
training_scores = zeros(N_images, 4);
features_train = features;
j = 1;
k = 1;


for i = 1 : T
    
    tic;
    
    if flag_hardnegative
        [error_1000, optimal_filter_error, optimal_filter_index, optimal_filter_params] = compute_errors_HN(Y, features_train, w);
    else
        [error_1000, optimal_filter_error, optimal_filter_index, optimal_filter_params] = compute_errors(Y, features_train, w);
    end
    
    alpha = 0.5 * log((1 - optimal_filter_error) / optimal_filter_error);
    z = 2 * sqrt(optimal_filter_error * (1 - optimal_filter_error));
    
    responses = compute_response(optimal_filter_params, optimal_filter_index, features_train);
    
    w = w .* exp(-Y .* responses .* alpha) ./ z;
    alpha_H(i) = alpha;
    param_H(i, :) = optimal_filter_params;
    index_H(i) = optimal_filter_index;
    
    if(i==1 || i==10 || i==50 || i==100 || i==200)
        error_weak(:, j) = error_1000;
        j = j + 1;
    end
    
    features_train(optimal_filter_index, :) = -200;
    
    [verify_scores, verify_Y] = strong_classifier_verify(alpha_H, param_H, index_H, features, i);
    error_tmp = zeros(N_images, 1);
    error_tmp(verify_Y ~= Y) = 1;
    training_error(i) = sum(error_tmp) / N_images;
    
    if(i==10 || i==50 || i==100 || i==200)
        training_scores(:, k) = verify_scores;
        k = k + 1;
    end
    
    fprintf('Boosting the %dth weak classifier took %.2f secs.\n', i, toc);
    
end


%% ============= Plotting region =============
set(0, 'defaultfigurecolor', [1 1 1]);
figure(1);
for i = 1 : 20
    subplot(4, 5, i);
    index_cur = index_H(i);
    white_box = filters{index_cur, 1};
    num_white_box = size(white_box, 1);
    black_box = filters{index_cur, 2};
    num_black_box = size(black_box, 1);
    hold on
    for j = 1 : num_white_box
        rectangle('Position', white_box(j, :), 'FaceColor', [1 1 1]);
    end
    for j = 1 : num_black_box
        rectangle('Position', black_box(j, :), 'FaceColor', [0 0 0]);
    end
    hold off
    axis([0 16 0 16]);
    titlestr = strcat( sprintf( '%d', i ), 'th Haar Filter' );
    title(titlestr, 'FontSize', 20);
end


figure(2);
x_axis = linspace(1, T, T);
plot(x_axis, training_error, 'LineWidth', 3);
xlabel('Number of training iterations', 'FontSize', 20);
ylabel('Training error', 'FontSize', 20);
title('Training Error of Strong Classifier (Adaboost)', 'FontSize', 20);
grid on


figure(3);
x_axis = linspace(1, 1000, 1000);
hold on
plot(x_axis, error_weak(:, 1), 'LineWidth', 3);
plot(x_axis, error_weak(:, 2), 'LineWidth', 3);
plot(x_axis, error_weak(:, 4), 'LineWidth', 3);
plot(x_axis, error_weak(:, 3), 'LineWidth', 3);
plot(x_axis, error_weak(:, 5), 'LineWidth', 3);
hold off
xlabel('Number of weak classifiers', 'FontSize', 20);
ylabel('Training error', 'FontSize', 20);
title('Training Error of Weak Classifiers (Adaboost)', 'FontSize', 20);
legend('T=1', 'T=10', 'T=50', 'T=100', 'T=200');
grid on


figure(4)
hold on
histogram(training_scores(1:N_pos, 1), 30, 'FaceColor', 'b');
histogram(training_scores(N_pos+1:end, 1), 30, 'FaceColor', 'r');
hold off
legend('Pos (face)', 'Neg (non-face)');
xlabel('Scores', 'FontSize', 20);
ylabel('Number', 'FontSize', 20);
title('Histograms when T=10', 'FontSize', 20);


figure(5)
hold on
histogram(training_scores(1:N_pos, 2), 30, 'FaceColor', 'b');
histogram(training_scores(N_pos+1:end, 2), 30, 'FaceColor', 'r');
hold off
legend('Pos (face)', 'Neg (non-face)');
xlabel('Scores', 'FontSize', 20);
ylabel('Number', 'FontSize', 20);
title('Histograms when T=50', 'FontSize', 20);


figure(6)
hold on
histogram(training_scores(1:N_pos, 3), 30, 'FaceColor', 'b');
histogram(training_scores(N_pos+1:end, 3), 30, 'FaceColor', 'r');
hold off
legend('Pos (face)', 'Neg (non-face)');
xlabel('Scores', 'FontSize', 20);
ylabel('Number', 'FontSize', 20);
title('Histograms when T=100', 'FontSize', 20);


figure(7)
hold on
histogram(training_scores(1:N_pos, 4), 30, 'FaceColor', 'b');
histogram(training_scores(N_pos+1:end, 4), 30, 'FaceColor', 'r');
hold off
legend('Pos (face)', 'Neg (non-face)');
xlabel('Scores', 'FontSize', 20);
ylabel('Number', 'FontSize', 20);
title('Histograms when T=200', 'FontSize', 20);


figure(8);
Y_tmp = ones(N_images, 1);
Y_tmp(Y==-1) = 0;
Y_tmp_2 = [Y_tmp Y_tmp Y_tmp Y_tmp]';
[tpr, fpr, ~] = roc(Y_tmp_2, training_scores');
hold on
plot(fpr{1}, tpr{1}, 'LineWidth', 3);
plot(fpr{2}, tpr{2}, 'LineWidth', 3);
plot(fpr{3}, tpr{3}, 'LineWidth', 3);
plot(fpr{4}, tpr{4}, 'LineWidth', 3);
hold off
legend('T=10', 'T=50', 'T=100', 'T=200');
xlabel('False Alarm Rate', 'FontSize', 20);
ylabel('Positive Alarm Rate', 'FontSize', 20);
title('ROC curves', 'FontSize', 20);


%% ============= Helper functions =============
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