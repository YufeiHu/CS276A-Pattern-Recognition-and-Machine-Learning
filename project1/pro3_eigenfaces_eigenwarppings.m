%% =================== File header ===================
% Title: CS 276A - Pattern Recognition and Machine Learning Project 1
% Subtitle: Problem 3
% Author: Yufei Hu
% Date: 10/21/2017

clc
clear all
close all
set(0, 'defaultfigurecolor', [1 1 1])


%% ========== Eigen-warpping and eigen-faces ==========
% User definition area
num_train = 150;
num_test = 27;
num_eigenfaces = 10;
num_eigenwarpping = 10;
data_path = './face_data_new/landmark_87/face';
image_data_path = './face_data_new/face/face';


% Derive parameters automatically
I_prob = imread( strcat( image_data_path, '000.bmp' ) );
[image_height, image_width] = size( I_prob );

L_prob = importdata( strcat( data_path, '000_87pt.dat' ) );
L_prob(1) = [];
[num_landmarks, tmp] = size( L_prob );
num_landmarks = num_landmarks / 2;


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

for i = 1 : num_train
    landmark_train_pos{i} = zeros(num_landmarks, 2);
    for j = 1 : num_landmarks
        landmark_train_pos{i}(j, 1) = landmark_train{i}(2*j-1);
        landmark_train_pos{i}(j, 2) = landmark_train{i}(2*j);
    end
end

for i = 1 : num_test
    landmark_test_pos{i} = zeros(num_landmarks, 2);
    for j = 1 : num_landmarks
        landmark_test_pos{i}(j, 1) = landmark_test{i}(2*j-1);
        landmark_test_pos{i}(j, 2) = landmark_test{i}(2*j);
    end
end

ML_pos = zeros(num_landmarks, 2);
for i = 1 : num_landmarks
    ML_pos(i, 1) = ML(2*i-1);
    ML_pos(i, 2) = ML(2*i);
end


% Align training and testing faces
Y_train_warp = [];
M_warp = zeros( image_height, image_width );
for i = 1 : num_train
    image_train_warp{i} = double( warpImage_new(image_train{i}, landmark_train_pos{i}, ML_pos) );
    Y_train_warp = [ Y_train_warp image_train_warp{i}(:) ];
    M_warp = M_warp + image_train_warp{i};
end
M_warp = M_warp ./ num_train;

Y_test_warp = [];
for i = 1 : num_test
    image_test_warp{i} = double( warpImage_new(image_test{i}, landmark_test_pos{i}, ML_pos) );
    Y_test_warp = [ Y_test_warp image_test_warp{i}(:) ];
end

figure(1)
for i = 1 : 25
    subplot(5, 5, i);
    imshow(image_train_warp{i}, []);
end
suptitle('Warpped training faces');

figure(2)
for i = 1 : 25
    subplot(5, 5, i);
    imshow(image_train{i}, []);
end
suptitle('Original training faces');


% Mean normalize all the training images
M_warp_vec = M_warp(:);
for i = 1 : num_train
    Y_train_warp_norm(:, i) = Y_train_warp(:, i) - M_warp_vec;
    L_train_norm(:, i) = L_train(:, i) - ML;
end

for i = 1 : num_test
    Y_test_warp_norm(:, i) = Y_test_warp(:, i) - M_warp_vec;
    L_test_norm(:, i) = L_test(:, i) - ML;
end


% Calculate eigenvectors (eigen-faces) and eigenvalues
% [U, S, V] = svds( Y_train_warp_norm, num_eigenfaces );
[U, V, S] = pca( Y_train_warp_norm' );
for i = 1 : num_eigenfaces
    eigenfaces{i} = reshape( U(:, i), image_height, image_width );
end


% Calculate eigenvectors (eigen-warpping) and eigenvalues
% [UL, SL, VL] = svds( L_train_norm, num_eigenwarpping );
[UL, VL, SL] = pca( L_train_norm' );
for i = 1 : num_eigenwarpping
    eigenwarpping{i} = zeros(num_landmarks, 2);
    for j = 1 : num_landmarks
        eigenwarpping{i}(j, 1) = UL(2*j-1, i) + ML(2*j-1);
        eigenwarpping{i}(j, 2) = UL(2*j, i) + ML(2*j);
    end
end


% Reconstruct all aligned testing images
Y_test_warp_norm_transpose = transpose( Y_test_warp_norm );
b = Y_test_warp_norm_transpose * U;
Y_test_warp_reconstruct = zeros( image_height * image_width, num_test );
for i = 1 : num_test
    for j = 1 : num_eigenfaces
        Y_test_warp_reconstruct(:, i) = Y_test_warp_reconstruct(:, i) + b(i, j) * U(:, j);
    end
    Y_test_warp_reconstruct(:, i) = Y_test_warp_reconstruct(:, i) + M_warp_vec;
    image_test_warp_reconstruct{i} = reshape(Y_test_warp_reconstruct(:, i), image_height, image_width);
end


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


% Warp all reconstructed testing images
for i = 1 : num_test
    image_test_reconstruct_2{i} = double( warpImage_new(image_test_warp_reconstruct{i}, ML_pos, landmark_test_reconstruct{i}) );
end


% Show all reconstructed testing images and original testing images
figure(3)
for i = 1 : num_test
    subplot(5, 6, i)
    imshow(image_test{i}, []);
end
suptitle('Original Testing Faces');

figure(4)
for i = 1 : num_test
    subplot(5, 6, i)
    imshow(image_test_reconstruct_2{i}, []);
end
suptitle('Reconstructed Testing Faces');


% Calculate and plot reconstruction errors
num_eigenfaces_max = 10;
error_2 = zeros(1, num_eigenfaces_max);
for i = 1 : num_eigenfaces_max
    i
    b = Y_test_warp_norm_transpose * U(:, 1:i);
    Y_test_reconstruct_mul = zeros( image_height * image_width, num_test );
    [tmp, num_eigenfaces_mul] = size( b );
    
    Y_test_warp_reconstruct_2 = [];
    for j = 1 : num_test
        for k = 1 : num_eigenfaces_mul
            Y_test_reconstruct_mul(:, j) = Y_test_reconstruct_mul(:, j) + b(j, k) * U(:, k);
        end
        Y_test_reconstruct_mul(:, j) = Y_test_reconstruct_mul(:, j) + M_warp_vec;
        image_test_warp_reconstruct_mul = reshape(Y_test_reconstruct_mul(:, j), image_height, image_width);
        image_test_warp_reconstruct_2{j} = double( warpImage_new(image_test_warp_reconstruct_mul, ML_pos, landmark_test_reconstruct{j}) );
        Y_test_warp_reconstruct_2 = [ Y_test_warp_reconstruct_2 image_test_warp_reconstruct_2{j}(:) ];
    end
    distances = ( Y_test_warp_reconstruct_2 - Y_test ) .^ 2;
    error_1{i} = sum(distances, 1) ./ (image_width * image_height);
    error_2(i) = sum( error_1{i} ) / num_test;
end

x_axis = linspace(1, num_eigenfaces_max, num_eigenfaces_max);
figure(5);
plot(x_axis, error_2, 'LineWidth', 3);
xlabel('Number of eigen-faces');
ylabel('Reconstruct Errors for testing faces');
grid on


% Calculate and plot more reconstruction errors
num_eigenfaces_max = 149;
error_2 = zeros(1, num_eigenfaces_max);
for i = 1 : num_eigenfaces_max
    i
    b = Y_test_warp_norm_transpose * U(:, 1:i);
    Y_test_reconstruct_mul = zeros( image_height * image_width, num_test );
    [tmp, num_eigenfaces_mul] = size( b );
    
    bL = L_test_norm_transpose * UL;
    L_test_reconstruct_mul = zeros( num_landmarks * 2, num_test );
    [tmp, num_eigenwarpping_mul] = size( bL );
    
    
    Y_test_warp_reconstruct_2 = [];
    for j = 1 : num_test
        
        for k = 1 : num_eigenfaces_mul
            L_test_reconstruct_mul(:, j) = L_test_reconstruct_mul(:, j) + bL(j, k) * UL(:, k);
            Y_test_reconstruct_mul(:, j) = Y_test_reconstruct_mul(:, j) + b(j, k) * U(:, k);
        end
        L_test_reconstruct_mul(:, j) = L_test_reconstruct_mul(:, j) + ML;
        
        for k = 1 : num_landmarks
            landmark_test_reconstruct_mul{j}(k, 1) = L_test_reconstruct_mul(2*k-1, j);
            landmark_test_reconstruct_mul{j}(k, 2) = L_test_reconstruct_mul(2*k, j);
        end
        
        Y_test_reconstruct_mul(:, j) = Y_test_reconstruct_mul(:, j) + M_warp_vec;
        
        image_test_warp_reconstruct_mul = reshape(Y_test_reconstruct_mul(:, j), image_height, image_width);
        image_test_warp_reconstruct_2{j} = double( warpImage_new(image_test_warp_reconstruct_mul, ML_pos, landmark_test_reconstruct{j}) );
        Y_test_warp_reconstruct_2 = [ Y_test_warp_reconstruct_2 image_test_warp_reconstruct_2{j}(:) ];
    end
    distances = ( Y_test_warp_reconstruct_2 - Y_test ) .^ 2;
    error_1{i} = sum(distances, 1) ./ (image_width * image_height);
    error_2(i) = sum( error_1{i} ) / num_test;
end

x_axis = linspace(1, num_eigenfaces_max, num_eigenfaces_max);
figure(6);
plot(x_axis, error_2, 'LineWidth', 3);
xlabel('Number of eigen-faces');
ylabel('Reconstruct Errors for testing faces');
grid on


% Save eigen-faces, mean face, eigen-warppings, mean landmark
save('data_for_pro4', 'M_warp', 'ML', 'ML_pos', 'S', 'U', 'SL', 'UL');