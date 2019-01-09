%% =================== File header ===================
% Title: CS 276A - Pattern Recognition and Machine Learning Project 4
% Subtitle: Problem 1 - 1
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
diary('P1_1.out');
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


% Cross validation
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
    [predict_label, ~, ~] = libsvmpredict(Y_test(:, t), X_test, model);
    predict_test(:, t) = predict_label;
    
end
fprintf('Training time: %.2f second\n', toc);


test_mse = zeros(num_model, 1);
for i = 1 : num_model
    test_mse(i) = sum((predict_test(:, i) - Y_test(:, i)) .^ 2) / num_test;
end


save('P1_1.mat', 'predict_test', 'train_mse', 'test_mse', 'param');


% Plot
figure(1);
hold on
plot(train_mse, 'LineWidth', 3);
plot(test_mse, 'LineWidth', 3);
plot(train_mse, 'b.', 'MarkerSize', 40);
plot(test_mse, 'r.' ,'MarkerSize', 40);
hold off
legend('Training mse', 'Testing mse');
xlabel('Model Index', 'FontSize', 20);
ylabel('MSE', 'FontSize', 20);
grid on