%% =================== File header ===================
% Title: CS 276A - Pattern Recognition and Machine Learning Project 1
% Subtitle: Problem 1
% Author: Yufei Hu
% Date: 10/19/2017

clc
clear all
close all
set(0, 'defaultfigurecolor', [1 1 1])


%% ========== Eigen-faces without landmark alignment ==========
% User definition area
num_train = 150;
num_test = 27;
num_eigenfaces = 20;
data_path = './face_data_new/face/face';


% Derive parameters automatically
I_prob = imread( strcat( data_path, '000.bmp' ) );
[image_height, image_width] = size( I_prob );


% Read training and testing images, calculate mean face
j = 1;
Y_train = [];
M = zeros( image_height, image_width );
for i = 1 : (num_train + 1)
    if(i ~= 104)
        image_path = strcat( data_path, sprintf( '%03d', i-1 ), '.bmp' );
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
    image_path = strcat( data_path, sprintf( '%03d', i-1 ), '.bmp' );
    image_test{j} = double( imread( image_path ) );
    Y_test = [ Y_test image_test{j}(:) ];
    j = j + 1;
end

figure(1)
imshow(M, []);
title('Mean Face', 'FontSize', 20);


% Mean normalize all the images
M_vec = M(:);
for i = 1 : num_train
    Y_train_norm(:, i) = Y_train(:, i) - M_vec;
end

for i = 1 : num_test
    Y_test_norm(:, i) = Y_test(:, i) - M_vec;
end


% Calculate eigenvectors (eigen-faces) and eigenvalues
% [U, S, V] = svds( cov( Y_train_norm' ), num_eigenfaces );
[U, V, S] = pca( Y_train_norm' );


for i = 1 : num_eigenfaces
    eigenfaces{i} = reshape( U(:, i), image_height, image_width );
end


% Show all the eigenfaces
for i = 1 : num_eigenfaces
    figure(2);
    subplot(4, num_eigenfaces / 4, i);
    imshow(eigenfaces{i}, []);
    titlestr = strcat( 'Face ', sprintf( '%d', i ) );
    title(titlestr, 'FontSize', 20);
end


% Reconstruct all testing images
Y_test_norm_transpose = transpose( Y_test_norm );
b = Y_test_norm_transpose * U;
Y_test_reconstruct = zeros( image_height * image_width, num_test );
for i = 1 : num_test
    for j = 1 : num_eigenfaces
        Y_test_reconstruct(:, i) = Y_test_reconstruct(:, i) + b(i, j) * U(:, j);
    end
    Y_test_reconstruct(:, i) = Y_test_reconstruct(:, i) + M_vec;
    image_test_reconstruct{i} = reshape(Y_test_reconstruct(:, i), image_height, image_width);
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
    imshow(image_test_reconstruct{i}, []);
end
suptitle('Reconstructed Testing Faces');


% Calculate and plot reconstruction errors
num_eigenfaces_max = 20;
error_2 = zeros(1, num_eigenfaces_max);
for i = 1 : num_eigenfaces_max
    
    b = Y_test_norm_transpose * U(:, 1:i);
    Y_test_reconstruct_mul = zeros( image_height * image_width, num_test );
    [tmp, num_eigenfaces_mul] = size( b );
    
    for j = 1 : num_test
        for k = 1 : num_eigenfaces_mul
            Y_test_reconstruct_mul(:, j) = Y_test_reconstruct_mul(:, j) + b(j, k) * U(:, k);
        end
        Y_test_reconstruct_mul(:, j) = Y_test_reconstruct_mul(:, j) + M_vec;
    end
    distances = ( Y_test_reconstruct_mul - Y_test ) .^ 2;
    error_1{i} = sum(distances, 1) ./ (image_width * image_height);
    error_2(i) = sum( error_1{i} ) / num_test;
end
x_axis = linspace(1, num_eigenfaces_max, num_eigenfaces_max);
figure(5);
plot(x_axis, error_2, 'LineWidth', 3);
xlabel('Number of eigen-faces');
ylabel('Reconstruct Errors for testing faces');
grid on


% Extra: Calculate and plot more reconstruction errors
num_eigenfaces_max = 149;
error_2 = zeros(1, num_eigenfaces_max);
for i = 1 : num_eigenfaces_max
    
    b = Y_test_norm_transpose * U(:, 1:i);
    Y_test_reconstruct_mul = zeros( image_height * image_width, num_test );
    [tmp, num_eigenfaces_mul] = size( b );
    
    for j = 1 : num_test
        for k = 1 : num_eigenfaces_mul
            Y_test_reconstruct_mul(:, j) = Y_test_reconstruct_mul(:, j) + b(j, k) * U(:, k);
        end
        Y_test_reconstruct_mul(:, j) = Y_test_reconstruct_mul(:, j) + M_vec;
    end
    distances = ( Y_test_reconstruct_mul - Y_test ) .^ 2;
    error_1{i} = sum(distances, 1) ./ (image_width * image_height);
    error_2(i) = sum( error_1{i} ) / num_test;
end
x_axis = linspace(1, num_eigenfaces_max, num_eigenfaces_max);
figure(6);
plot(x_axis, error_2, 'LineWidth', 3);
xlabel('Number of eigen-faces');
ylabel('Reconstruct Errors for testing faces');
grid on