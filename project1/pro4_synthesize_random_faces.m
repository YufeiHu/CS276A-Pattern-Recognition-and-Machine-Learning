%% =================== File header ===================
% Title: CS 276A - Pattern Recognition and Machine Learning Project 1
% Subtitle: Problem 4
% Author: Yufei Hu
% Date: 10/21/2017

clc
clear all
close all
set(0, 'defaultfigurecolor', [1 1 1])


%% ========== Synthesize random faces ==========
% User definition area
load('data_for_pro4.mat');
num_random_faces = 20;


% Generate random parameters
[num_landmarks, tmp] = size(ML_pos);
[image_width, image_height] = size(M_warp);
[num_eigenfaces, tmp] = size(S);
[num_eigenwarpping, tmp] = size(SL);

a = zeros(num_random_faces, num_eigenfaces);
b = zeros(num_random_faces, num_eigenwarpping);
for i = 1 : num_random_faces
    
    for j = 1 : num_eigenfaces
        lamda = sqrt( S(j) );
        a(i, j) = lamda .* randn();
    end
    
    for j = 1 : num_eigenwarpping
        lamda = sqrt( SL(j) );
        b(i, j) = lamda .* randn();
    end
    
end


% Construct random faces
random_faces_1 = zeros( image_height * image_width, num_random_faces );
for i = 1 : num_random_faces
    for j = 1 : num_eigenfaces
        random_faces_1(:, i) = random_faces_1(:, i) + a(i, j) * U(:, j);
    end
    random_faces_1(:, i) = random_faces_1(:, i) + M_warp(:);
    random_faces_2{i} = reshape(random_faces_1(:, i), image_width, image_height);
end

% Construct random landmarks
random_landmarks_1 = zeros( num_landmarks * 2, num_random_faces );
for i = 1 : num_random_faces
    for j = 1 : num_eigenwarpping
        random_landmarks_1(:, i) = random_landmarks_1(:, i) + b(i, j) * UL(:, j);
    end
    random_landmarks_1(:, i) = random_landmarks_1(:, i) + ML;
    for k = 1 : num_landmarks
        random_landmarks_2{i}(k, 1) = random_landmarks_1(2*k-1, i);
        random_landmarks_2{i}(k, 2) = random_landmarks_1(2*k, i);
    end
end


% Generate random faces
for i = 1 : num_random_faces
    random_faces_3{i} = warpImage_new(random_faces_2{i}, ML_pos, random_landmarks_2{i});
end


% Show these random faces
figure(1)
for i = 1 : num_random_faces
    subplot(4, num_random_faces/4, i)
    imshow(random_faces_3{i}, []);
end
suptitle('Random Faces');