%% =================== File header ===================
% Title: CS 276A - Pattern Recognition and Machine Learning Project 4
% Subtitle: Problem 1 - 2
% Author: Yufei Hu
% Date: 12/02/2017


clear
close all;
set(0, 'defaultfigurecolor', [1 1 1]);


% Switches
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
diary('P1_2.out');
rng(123);
addpath('libsvm-3.21/matlab');


% Data
disp('loading data ...');
load('train-anno.mat', 'face_landmark', 'trait_annotation');
features = face_landmark;
labels = trait_annotation;
clear face_landmark;
clear trait_annotation;
clear flag_compile_libsvm_c;
clear flag_compile_libsvm_mex;



%% Perform k-fold cross-validation
% User definition
num_fold = 5;
train_ratio = 0.8;
[C, gamma, epsilon] = meshgrid(-5:2:13, -17:2:-1, -9:2:-5);


% Preprocess the data
[features_clean, ~] = clean_features(features);
features_clean(:, 77:78) = [];
[features_rescale] = rescale_features(features_clean);


% Compute mean landmarks
[ML] = compute_mean_landmark(features_clean);
[ML_pos] = adjust_landmark(ML);


% Extract HoG features
features_HoG = [];
for i = 1 : size(features_rescale, 1)
    fprintf('HoG iteration: %d\n', i);
    img_path = ['./img/M', sprintf('%04d', i), '.jpg'];
    im = imread(img_path);
    [landmark_adjust] = adjust_landmark(features_clean(i, :));
    im = warp_image(im, landmark_adjust, ML_pos);
    im = double(im);
    hogfeat = HoGfeatures(im);
    hogfeat = double(hogfeat);
    hogfeat = reshape(hogfeat(:,:,16), 1, []);
    features_HoG = [features_HoG hogfeat(1771:1870)'];
end
features_HoG = features_HoG';
[features_HoG] = rescale_HoG_features(features_HoG);


% Cross validation
features_rescale = [features_rescale features_HoG];
num_total = size(features_rescale, 1);
num_train = floor(num_total * train_ratio);
num_test = num_total - num_train;
X_train = features_rescale(1:num_train, :);
Y_train = labels(1:num_train, :);
X_test = features_rescale((num_train+1):end, :);
Y_test = labels((num_train+1):end, :);

num_model = 14;
train_mse = zeros(num_model, 1);
param = zeros(num_model, 3);
predict_test = zeros(num_test, num_model);

model_firstlayer = cell(num_model, 1);
tic;
for t = 1 : num_model
    
    fprintf('Iteration: %d', t);
    acc = zeros(numel(C), 1);
    
    for i = 1 : numel(C)
        cmd = ['-s 3 ', sprintf('-c %f -g %f -p %f -v %d -q', 2^C(i), 2^gamma(i), 2^epsilon(i), num_fold)];
        acc(i) = libsvmtrain(Y_train(:, t), X_train, cmd);
    end
    
    [~, min_idx] = min(acc);
    train_mse(t) = acc(min_idx);
    param(t, 1) = C(min_idx);
    param(t, 2) = gamma(min_idx);
    param(t, 3) = epsilon(min_idx);
    
    cmd = ['-s 3 ', sprintf('-c %f -g %f -p %f -q', 2^C(min_idx), 2^gamma(min_idx), 2^epsilon(min_idx))];
    model = libsvmtrain(Y_train(:, t), X_train, cmd);
    model_firstlayer{t} = libsvmtrain(labels(:, t), features_rescale, cmd);
    [predict_label, ~, ~] = libsvmpredict(Y_test(:, t), X_test, model);
    predict_test(:, t) = predict_label;
    
end
fprintf('Training time: %.2f second\n', toc);


test_mse = zeros(num_model, 1);
for i = 1 : num_model
    test_mse(i) = sum((predict_test(:, i) - Y_test(:, i)) .^ 2) / num_test;
end


save('model_firstlayer.mat', 'model_firstlayer');
% save('P1_2.mat', 'predict_test', 'train_mse', 'test_mse', 'param');


% Plot
P1_1 = load('P1_1.mat');
train_mse_poor = P1_1.train_mse;
test_mse_poor = P1_1.test_mse;
figure(1);
hold on
plot(train_mse, '-bo', 'LineWidth', 3);
plot(test_mse, '-ro', 'LineWidth', 3);
plot(train_mse_poor, ':b*', 'LineWidth', 3);
plot(test_mse_poor, ':r*', 'LineWidth', 3);
hold off
legend('Training mse (Richer)', 'Testing mse (Richer)', 'Training mse (Poorer)', 'Testing mse (Poorer)');
xlabel('Model Index', 'FontSize', 20);
ylabel('MSE', 'FontSize', 20);
grid on