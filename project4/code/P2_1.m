%% =================== File header ===================
% Title: CS 276A - Pattern Recognition and Machine Learning Project 4
% Subtitle: Problem 2 - 1
% Author: Yufei Hu
% Date: 12/02/2017


clear
close all;
set(0, 'defaultfigurecolor', [1 1 1]);


% Switches
flag_extract_HoG = 0;
flag_compile_libsvm_c = 0;
flag_compile_libsvm_mex = 0;


% Compile libsvm
if flag_compile_libsvm_c
    parent = cd('libsvm-3.21');
    [status, cmdout] = system('make');
    cd(parent);
    disp(status);
    disp(cmdout);
end

if flag_compile_libsvm_mex
    parent = cd('libsvm-3.21/matlab');
    make;
    cd(parent);
end


% Setup
diary('P2_1.out');
rng(123);
addpath('libsvm-3.21/matlab');


% Data
disp('loading data ...');
data_sen = load('stat-sen.mat', 'face_landmark', 'vote_diff');
data_gov = load('stat-gov.mat', 'face_landmark', 'vote_diff');
sen_landmark = data_sen.face_landmark;
sen_votediff = data_sen.vote_diff;
gov_landmark = data_gov.face_landmark;
gov_votediff = data_gov.vote_diff;

clear flag_compile_libsvm_c;
clear flag_compile_libsvm_mex;
clear data_sen;
clear data_gov;


%% Perform k-fold cross-validation
% User definition
num_fold = 5;
train_ratio = 0.7;
C = -15:1:15;


% Preprocess the data
[gov_landmark, ~] = clean_features(gov_landmark);
[sen_landmark, ~] = clean_features(sen_landmark);
gov_landmark(:, 77:78) = [];
sen_landmark(:, 77:78) = [];
[gov_landmark_rescale] = rescale_features(gov_landmark);
[sen_landmark_rescale] = rescale_features(sen_landmark);


% Compute mean landmarks
[gov_ML] = compute_mean_landmark(gov_landmark);
[sen_ML] = compute_mean_landmark(sen_landmark);
[gov_ML_pos] = adjust_landmark(gov_ML);
[sen_ML_pos] = adjust_landmark(sen_ML);


% Extract HoG features
if(flag_extract_HoG==0)
    load('P2_1_tmp.mat');
else
    gov_HoG = [];
    for i = 1 : size(gov_landmark, 1)
        fprintf('Governor HoG iteration: %d\n', i);
        img_path = ['./img-elec/governor/G', sprintf('%04d', i), '.jpg'];
        im = imread(img_path);
        [landmark_adjust] = adjust_landmark(gov_landmark(i, :));
        im = warp_image(im, landmark_adjust, gov_ML_pos);
        im = double(im);
        hogfeat = HoGfeatures(im);
        hogfeat = double(hogfeat);
        hogfeat = reshape(hogfeat(:,:,16), 1, []);
        gov_HoG = [gov_HoG hogfeat(1771:1900)'];
    end
    gov_HoG = gov_HoG';
    [gov_HoG] = rescale_HoG_features(gov_HoG);

    sen_HoG = [];
    for i = 1 : size(sen_landmark, 1)
        fprintf('Senator HoG iteration: %d\n', i);
        img_path = ['./img-elec/senator/S', sprintf('%04d', i), '.jpg'];
        im = imread(img_path);
        [landmark_adjust] = adjust_landmark(sen_landmark(i, :));
        im = warp_image(im, landmark_adjust, sen_ML_pos);
        im = double(im);
        hogfeat = HoGfeatures(im);
        hogfeat = double(hogfeat);
        hogfeat = reshape(hogfeat(:,:,16), 1, []);
        sen_HoG = [sen_HoG hogfeat(1771:1900)'];
    end
    sen_HoG = sen_HoG';
    [sen_HoG] = rescale_HoG_features(sen_HoG);
    
    save('P2_1_tmp.mat', 'gov_HoG', 'sen_HoG');
end

sen_HoG = sen_HoG(:, 121:130);

% Cross validation (Governor)
gov_landmark_rescale = [gov_landmark_rescale gov_HoG];
gov_features = gov_landmark_rescale; % [gov_features] = compute_diff_features(gov_landmark_rescale);
[gov_labels] = compute_diff_labels(gov_votediff);

num_total = size(gov_labels, 1);
num_train = floor(num_total * train_ratio);
num_test = num_total - num_train;

train_votediff = gov_votediff(1:(2*num_train), :);
X_train = gov_features(1:(2*num_train), :);
Y_train = compute_diff_labels(train_votediff);

test_votediff = gov_votediff((2*num_train+1):end, :);
X_test = gov_features((2*num_train+1):end, :);
Y_test = compute_diff_labels(test_votediff);


acc_train = zeros(numel(C), 1);
acc_test = zeros(numel(C), 1);
indices = crossvalind('Kfold', Y_train(:, 1), num_fold);


for i = 1 : numel(C)
    
    %fprintf('Governor C iteration: %d\n', i);
    fold_train_acc = zeros(num_fold, 1);
    fold_test_acc = zeros(num_fold, 1);

    for k = 1 : num_fold
        
        fold_test_index = (indices == k);
        fold_train_index = ~fold_test_index;
        
        fold_test_index_features = zeros(size(fold_test_index, 1) * 2, 1);
        for j = 1 : size(fold_test_index, 1)
            if(fold_test_index(j) == 1)
                fold_test_index_features(2*j-1) = 1;
                fold_test_index_features(2*j) = 1;
            end
        end
        
        fold_train_index_features = zeros(size(fold_train_index, 1) * 2, 1);
        for j = 1 : size(fold_train_index, 1)
            if(fold_train_index(j) == 1)
                fold_train_index_features(2*j-1) = 1;
                fold_train_index_features(2*j) = 1;
            end
        end

        fold_num_train = sum(fold_train_index);
        fold_num_test = sum(fold_test_index);
        
        fold_train_votediff = train_votediff(fold_train_index_features==1, :);
        fold_X_train = X_train(fold_train_index_features==1, :);
        fold_Y_train = compute_diff_labels(fold_train_votediff);
        
        fold_test_votediff = train_votediff(fold_test_index_features==1, :);
        fold_X_test = X_train(fold_test_index_features==1, :);
        fold_Y_test = compute_diff_labels(fold_test_votediff);
        
        w = ranksvm(fold_X_train, fold_Y_train, ones(size(fold_Y_train, 1), 1) .* (2^C(i)));
        
        fold_Y_test_hat = sign((fold_X_test([1:size(fold_Y_test, 1)]*2, :) - fold_X_test([1:size(fold_Y_test, 1)]*2-1, :))*w);
        fold_test_acc(k) = sum(fold_Y_test_hat==1) / size(fold_Y_test, 1);
        
        fold_Y_train_hat = sign((fold_X_train([1:size(fold_Y_train, 1)]*2, :) - fold_X_train([1:size(fold_Y_train, 1)]*2-1, :))*w);
        fold_train_acc(k) = sum(fold_Y_train_hat==1) / size(fold_Y_train, 1);
        
    end
    
    acc_train(i) = sum(fold_train_acc) / num_fold;
    acc_test(i) = sum(fold_test_acc) / num_fold;
    
end

[~, max_idx] = max(acc_test);
gov_param = C(max_idx);

w = ranksvm(X_train, Y_train, ones(size(Y_train, 1), 1) .* (2^gov_param));

Y_test_hat = sign((X_test([1:size(Y_test, 1)]*2, :) - X_test([1:size(Y_test, 1)]*2-1, :))*w);
gov_test_acc = sum(Y_test_hat==1) / size(Y_test, 1);

Y_train_hat = sign((X_train([1:size(Y_train, 1)]*2, :) - X_train([1:size(Y_train, 1)]*2-1, :))*w);
gov_train_acc = sum(Y_train_hat==1) / size(Y_train, 1);


% Cross validation (Senator)
sen_landmark_rescale = [sen_landmark_rescale sen_HoG];
sen_features = sen_landmark_rescale; % [sen_features] = compute_diff_features(sen_landmark_rescale);
[sen_labels] = compute_diff_labels(sen_votediff);

num_total = size(sen_labels, 1);
num_train = floor(num_total * train_ratio);
num_test = num_total - num_train;

train_votediff = sen_votediff(1:(2*num_train), :);
X_train = sen_features(1:(2*num_train), :);
Y_train = compute_diff_labels(train_votediff);

test_votediff = sen_votediff((2*num_train+1):end, :);
X_test = sen_features((2*num_train+1):end, :);
Y_test = compute_diff_labels(test_votediff);


acc_train = zeros(numel(C), 1);
acc_test = zeros(numel(C), 1);
indices = crossvalind('Kfold', Y_train(:, 1), num_fold);


for i = 1 : numel(C)
    
    %fprintf('Senator C iteration: %d\n', i);
    fold_train_acc = zeros(num_fold, 1);
    fold_test_acc = zeros(num_fold, 1);

    for k = 1 : num_fold
        
        fold_test_index = (indices == k);
        fold_train_index = ~fold_test_index;
        
        fold_test_index_features = zeros(size(fold_test_index, 1) * 2, 1);
        for j = 1 : size(fold_test_index, 1)
            if(fold_test_index(j) == 1)
                fold_test_index_features(2*j-1) = 1;
                fold_test_index_features(2*j) = 1;
            end
        end
        
        fold_train_index_features = zeros(size(fold_train_index, 1) * 2, 1);
        for j = 1 : size(fold_train_index, 1)
            if(fold_train_index(j) == 1)
                fold_train_index_features(2*j-1) = 1;
                fold_train_index_features(2*j) = 1;
            end
        end

        fold_num_train = sum(fold_train_index);
        fold_num_test = sum(fold_test_index);
        
        fold_train_votediff = train_votediff(fold_train_index_features==1, :);
        fold_X_train = X_train(fold_train_index_features==1, :);
        fold_Y_train = compute_diff_labels(fold_train_votediff);
        
        fold_test_votediff = train_votediff(fold_test_index_features==1, :);
        fold_X_test = X_train(fold_test_index_features==1, :);
        fold_Y_test = compute_diff_labels(fold_test_votediff);
        
        w = ranksvm(fold_X_train, fold_Y_train, ones(size(fold_Y_train, 1), 1) .* (2^C(i)));
        
        fold_Y_test_hat = sign((fold_X_test([1:size(fold_Y_test, 1)]*2, :) - fold_X_test([1:size(fold_Y_test, 1)]*2-1, :))*w);
        fold_test_acc(k) = sum(fold_Y_test_hat==1) / size(fold_Y_test, 1);
        
        fold_Y_train_hat = sign((fold_X_train([1:size(fold_Y_train, 1)]*2, :) - fold_X_train([1:size(fold_Y_train, 1)]*2-1, :))*w);
        fold_train_acc(k) = sum(fold_Y_train_hat==1) / size(fold_Y_train, 1);
        
    end
    
    acc_train(i) = sum(fold_train_acc) / num_fold;
    acc_test(i) = sum(fold_test_acc) / num_fold;
    
end

[~, max_idx] = max(acc_test);
sen_param = C(max_idx);

w = ranksvm(X_train, Y_train, ones(size(Y_train, 1), 1) .* (2^sen_param));

Y_test_hat = sign((X_test([1:size(Y_test, 1)]*2, :) - X_test([1:size(Y_test, 1)]*2-1, :))*w);
sen_test_acc = sum(Y_test_hat==1) / size(Y_test, 1);

Y_train_hat = sign((X_train([1:size(Y_train, 1)]*2, :) - X_train([1:size(Y_train, 1)]*2-1, :))*w);
sen_train_acc = sum(Y_train_hat==1) / size(Y_train, 1);


% Save results
save('P2_1.mat', 'sen_param', 'sen_test_acc', 'sen_train_acc', 'gov_param', 'gov_test_acc', 'gov_train_acc');