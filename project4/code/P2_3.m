%% =================== File header ===================
% Title: CS 276A - Pattern Recognition and Machine Learning Project 4
% Subtitle: Problem 2 - 3
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
diary('P2_2.out');
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


%% Extract facial attributes
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
        gov_HoG = [gov_HoG hogfeat(1771:1870)'];
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
        sen_HoG = [sen_HoG hogfeat(1771:1870)'];
    end
    sen_HoG = sen_HoG';
    [sen_HoG] = rescale_HoG_features(sen_HoG);
    
    save('P2_1_tmp.mat', 'gov_HoG', 'sen_HoG');
end

gov_HoG = gov_HoG(:, 1:100);
sen_HoG = sen_HoG(:, 1:100);


% Append features
gov_landmark_rescale = [gov_landmark_rescale gov_HoG];
sen_landmark_rescale = [sen_landmark_rescale sen_HoG];


% Extract features
load('model_firstlayer.mat');
num_model = 14;
gov_features = zeros(size(gov_landmark_rescale, 1), num_model);
sen_features = zeros(size(sen_landmark_rescale, 1), num_model);
for i = 1 : num_model
    cur_model = model_firstlayer{i};
    [gov_features(:, i), ~, ~] = libsvmpredict(ones(size(gov_landmark_rescale, 1), 1), gov_landmark_rescale, cur_model);
    [sen_features(:, i), ~, ~] = libsvmpredict(ones(size(sen_landmark_rescale, 1), 1), sen_landmark_rescale, cur_model);
end


%% Compute correlations
gov_correlation = zeros(num_model, 1);
sen_correlation = zeros(num_model, 1);

for i = 1 : num_model
    gov_correlation(i) = corr(gov_features(:, i), gov_votediff);
    sen_correlation(i) = corr(sen_features(:, i), sen_votediff);
end

spider_labels = cell(num_model, 1);
spider_labels{1} = ['Old'];
spider_labels{2} = ['Masculine'];
spider_labels{3} = ['Baby-faced'];
spider_labels{4} = ['Competent'];
spider_labels{5} = ['Attractive'];
spider_labels{6} = ['Energetic'];
spider_labels{7} = ['Well-groomed'];
spider_labels{8} = ['Intelligent'];
spider_labels{9} = ['Honest'];
spider_labels{10} = ['Generous'];
spider_labels{11} = ['Trustworthy'];
spider_labels{12} = ['Confident'];
spider_labels{13} = ['Rich'];
spider_labels{14} = ['Dominant'];

axes_interval = 2; 
axes_precision = 4;
P = [gov_correlation sen_correlation]';
figure(1)
spider_plot(P, spider_labels, axes_interval, axes_precision,...
    'Marker', 'o',...
    'LineStyle', '-',...
    'LineWidth', 2,...
    'MarkerSize', 3);
legend('Governor', 'Senator');


% Extra
[gov_labels] = compute_diff_labels(gov_votediff);
[sen_labels] = compute_diff_labels(sen_votediff);
gov_w = ranksvm(gov_features, gov_labels, ones(size(gov_labels, 1), 1) .* (2^10));
sen_w = ranksvm(sen_features, sen_labels, ones(size(sen_labels, 1), 1) .* (2^(-1)));

axes_interval = 2; 
axes_precision = 4;
P = [gov_w sen_w]';
figure(2)
spider_plot(P, spider_labels, axes_interval, axes_precision,...
    'Marker', 'o',...
    'LineStyle', '-',...
    'LineWidth', 2,...
    'MarkerSize', 3);
legend('Governor', 'Senator');