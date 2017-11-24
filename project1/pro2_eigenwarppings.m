%% =================== File header ===================
% Title: CS 276A - Pattern Recognition and Machine Learning Project 1
% Subtitle: Problem 2
% Author: Yufei Hu
% Date: 10/21/2017

clc
clear all
close all
set(0, 'defaultfigurecolor', [1 1 1])


%% ========== Eigen-warpping of face landmarks ==========
% User definition area
num_train = 150;
num_test = 27;
num_eigenwarpping = 5;
data_path = './face_data_new/landmark_87/face';
image_data_path = './face_data_new/face/face';


% Read training and testing images
I_prob = imread( strcat( image_data_path, '000.bmp' ) );
[image_height, image_width] = size( I_prob );

j = 1;
Y_train = [];
M = zeros( image_height, image_width );
for i = 1 : (num_train + 1)
    if(i ~= 104)
        image_path = strcat( image_data_path, sprintf( '%03d', i-1 ), '.bmp' );
        image_train{j} = double( imread( image_path ) );
        Y_train = [ Y_train image_train{j}(:) ];
        M = M + image_train{j};
        j = j + 1;
    end
end
M = M ./ num_train;

j = 1;
Y_test = [];
for i = (num_train + 2) : (num_train + num_test + 1)
    image_path = strcat( image_data_path, sprintf( '%03d', i-1 ), '.bmp' );
    image_test{j} = double( imread( image_path ) );
    Y_test = [ Y_test image_test{j}(:) ];
    j = j + 1;
end


% Derive parameters automatically
L_prob = importdata( strcat( data_path, '000_87pt.dat' ) );
L_prob(1) = [];
[num_landmarks, tmp] = size( L_prob );
num_landmarks = num_landmarks / 2;


% Read training and testing landmarks, calculate mean landmark
j = 1;
L_train = [];
ML = zeros( num_landmarks * 2, 1 );
for i = 1 : (num_train + 1)
    if(i ~= 104)
        landmark_path = strcat( data_path, sprintf( '%03d', i-1 ), '_87pt.dat' );
        landmark_train{j} = double( importdata( landmark_path ) );
        landmark_train{j}(1) = [];
        L_train = [ L_train landmark_train{j}(:) ];
        ML = ML + landmark_train{j};
        j = j + 1;
    end
end
ML = ML ./ num_train;

j = 1;
L_test = [];
for i = (num_train + 2) : (num_train + num_test + 1)
    landmark_path = strcat( data_path, sprintf( '%03d', i-1 ), '_87pt.dat' );
    landmark_test{j} = double( importdata( landmark_path ) );
    landmark_test{j}(1) = [];
    L_test = [ L_test landmark_test{j}(:) ];
    j = j + 1;
end

ML_pos = zeros(num_landmarks, 2);
landmark_train_1 = zeros(num_landmarks, 2);
for i = 1 : num_landmarks
    ML_pos(i, 1) = ML(2*i-1);
    ML_pos(i, 2) = ML(2*i);
    landmark_train_1(i, 1) = landmark_train{36}(2*i-1);
    landmark_train_1(i, 2) = landmark_train{36}(2*i);
end

figure(1)
imshow(M, []);
hold on
plot(ML_pos(:, 1), ML_pos(:, 2), 'r.', 'MarkerSize', 25);
hold off
title('Mean Landmarks on Mean Face', 'FontSize', 20);

figure(2)
imshow(image_train{36}, []);
hold on
plot(ML_pos(:, 1), ML_pos(:, 2), 'r.', 'MarkerSize', 25);
plot(landmark_train_1(:, 1), landmark_train_1(:, 2), 'b.', 'MarkerSize', 25);
hold off
title('Mean Landmarks on Face 35', 'FontSize', 20);


% Mean normalize all the images
for i = 1 : num_train
    L_train_norm(:, i) = L_train(:, i) - ML;
end

for i = 1 : num_test
    L_test_norm(:, i) = L_test(:, i) - ML;
end


% Calculate eigenvectors (eigen-warpping) and eigenvalues
% [UL, SL, VL] = svds( L_train_norm, num_eigenwarpping );
[UL, VL, SL] = pca( L_train_norm' );
for i = 1 : num_eigenwarpping
    eigenwarpping{i} = zeros(num_landmarks, 2);
    eigenwarpping_diff{i} = zeros(num_landmarks, 2);
    for j = 1 : num_landmarks
        eigenwarpping{i}(j, 1) = UL(2*j-1, i) + ML(2*j-1);
        eigenwarpping{i}(j, 2) = UL(2*j, i) + ML(2*j);
        eigenwarpping_diff{i}(j, 1) = sqrt( SL(i) ) * UL(2*j-1, i) + ML(2*j-1);
        eigenwarpping_diff{i}(j, 2) = sqrt( SL(i) ) * UL(2*j, i) + ML(2*j);
    end
end


% Show all the eigen-warppings
figure(3);
for i = 1 : num_eigenwarpping
    subplot(2, 3, i);
    imshow(M, []);
    hold on
    plot(eigenwarpping{i}(:, 1), eigenwarpping{i}(:, 2), 'r.', 'MarkerSize', 15);
    hold off
    titlestr = strcat( 'Eigen-warpping   ', sprintf( ' %d', i ) );
    title(titlestr, 'FontSize', 20);
end
suptitle('Eigen-warppings on Mean Face');


figure(4);
imshow(M, []);
hold on
plot(eigenwarpping{1}(:, 1), eigenwarpping_diff{1}(:, 2), 'r.', 'MarkerSize', 30);
plot(eigenwarpping{2}(:, 1), eigenwarpping_diff{2}(:, 2), 'b.', 'MarkerSize', 30);
plot(eigenwarpping{3}(:, 1), eigenwarpping_diff{3}(:, 2), 'k.', 'MarkerSize', 30);
plot(eigenwarpping{4}(:, 1), eigenwarpping_diff{4}(:, 2), 'y.', 'MarkerSize', 30);
plot(eigenwarpping{5}(:, 1), eigenwarpping_diff{5}(:, 2), 'g.', 'MarkerSize', 30);
hold off
title('All Eigen-warppings on Mean Face', 'FontSize', 20);
legend('Eigen-warpping 1', 'Eigen-warpping 2', 'Eigen-warpping 3', 'Eigen-warpping 4', 'Eigen-warpping 5');


% Reconstruct all testing landmarks
L_test_norm_transpose = transpose( L_test_norm );
bL = L_test_norm_transpose * UL;
L_test_reconstruct = zeros( num_landmarks * 2, num_test );
for i = 1 : num_test
    for j = 1 : num_eigenwarpping
        L_test_reconstruct(:, i) = L_test_reconstruct(:, i) + bL(i, j) * UL(:, j);
    end
    L_test_reconstruct(:, i) = L_test_reconstruct(:, i) + ML;
    for k = 1 : num_landmarks
        landmark_test_reconstruct{i}(k, 1) = L_test_reconstruct(2*k-1, i);
        landmark_test_reconstruct{i}(k, 2) = L_test_reconstruct(2*k, i);
    end
end


% Show all reconstructed testing landmarks and original testing landmarks
for i = 1 : num_test
    landmark_test_or{i} = zeros(num_landmarks, 2);
    for j = 1 : num_landmarks
        landmark_test_or{i}(j, 1) = landmark_test{i}(2*j-1);
        landmark_test_or{i}(j, 2) = landmark_test{i}(2*j);
    end
end

figure(5)
for i = 1 : num_test
    subplot(5, 6, i)
    imshow(image_test{i}, []);
    hold on
    plot(landmark_test_reconstruct{i}(:, 1), landmark_test_reconstruct{i}(:, 2), 'r.', 'MarkerSize', 10);
    plot(landmark_test_or{i}(:, 1), landmark_test_or{i}(:, 2), 'b.', 'MarkerSize', 10);
    hold off
end
legend('Reconstructed Landmarks', 'Original Landmarks');
suptitle('Reconstructed Landmarks');


% Calculate and plot reconstruction errors
num_eigenwarpping_max = 5;
error_2_L = zeros(1, num_eigenwarpping_max);
for i = 1 : num_eigenwarpping_max
    
    bL = L_test_norm_transpose * UL(:, 1:i);
    L_test_reconstruct_mul = zeros( 2 * num_landmarks, num_test );
    [tmp, num_eigenwarpping_mul] = size( bL );
    
    for j = 1 : num_test
        for k = 1 : num_eigenwarpping_mul
            L_test_reconstruct_mul(:, j) = L_test_reconstruct_mul(:, j) + bL(j, k) * UL(:, k);
        end
        L_test_reconstruct_mul(:, j) = L_test_reconstruct_mul(:, j) + ML;
    end
    distances = ( L_test_reconstruct_mul - L_test ) .^ 2;
    error_1_L{i} = sqrt( sum(distances, 1) );
    error_2_L(i) = sum( error_1_L{i} ) / num_test;
end
x_axis = linspace(1, num_eigenwarpping_max, num_eigenwarpping_max);
figure(6);
plot(x_axis, error_2_L, 'LineWidth', 3);
xlabel('Number of eigen-warppings');
ylabel('Reconstruction Errors for testing landmarks');
grid on


% Extra: Calculate and plot more reconstruction errors
num_eigenwarpping_max = 149;
% [UL, SL, VL] = svds( L_train_norm, num_eigenwarpping_max );
error_2_L = zeros(1, num_eigenwarpping_max);
for i = 1 : num_eigenwarpping_max
    
    bL = L_test_norm_transpose * UL(:, 1:i);
    L_test_reconstruct_mul = zeros( 2 * num_landmarks, num_test );
    [tmp, num_eigenwarpping_mul] = size( bL );
    
    for j = 1 : num_test
        for k = 1 : num_eigenwarpping_mul
            L_test_reconstruct_mul(:, j) = L_test_reconstruct_mul(:, j) + bL(j, k) * UL(:, k);
        end
        L_test_reconstruct_mul(:, j) = L_test_reconstruct_mul(:, j) + ML;
    end
    distances = ( L_test_reconstruct_mul - L_test ) .^ 2;
    error_1_L{i} = sqrt( sum(distances, 1) );
    error_2_L(i) = sum( error_1_L{i} ) / num_test;
end
x_axis = linspace(1, num_eigenwarpping_max, num_eigenwarpping_max);
figure(7);
plot(x_axis, error_2_L, 'LineWidth', 3);
xlabel('Number of eigen-warppings');
ylabel('Reconstruction Errors for testing landmarks');
grid on