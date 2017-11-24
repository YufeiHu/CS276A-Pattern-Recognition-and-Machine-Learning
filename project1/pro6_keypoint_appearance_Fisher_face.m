%% =================== File header ===================
% Title: CS 276A - Pattern Recognition and Machine Learning Project 1
% Subtitle: Problem 6
% Author: Yufei Hu
% Date: 10/22/2017

clc
clear all
close all
set(0, 'defaultfigurecolor', [1 1 1])


%% ======== Find Fisher face for key point and appearance ========
% User definition area
num_male = 88;
num_female = 85;

image_height = 256;
image_width = 256;
num_landmarks = 87;

male_face_path = './face_data_new/male_face/face';
female_face_path = './face_data_new/female_face/face';
male_landmark_path = './face_data_new/male_landmark_87/face';
female_landmark_path = './face_data_new/female_landmark_87/face';


% Read images, calculate mean face
j = 1;
male_face = [];
male_face_mean = zeros( image_height, image_width );
for i = 1 : (num_male + 1)
    if(i ~= 58)
        image_path = strcat( male_face_path, sprintf( '%03d', i-1 ), '.bmp' );
        male_face_image{j} = double( imread( image_path ) );
        male_face = [ male_face male_face_image{j}(:) ];
        male_face_mean = male_face_mean + male_face_image{j};
        j = j + 1;
    end
end
male_face_mean = male_face_mean ./ num_male;
male_face_mean_vec = male_face_mean(:);


j = 1;
female_face = [];
female_face_mean = zeros( image_height, image_width );
for i = 1 : (num_female)
    image_path = strcat( female_face_path, sprintf( '%03d', i-1 ), '.bmp' );
    female_face_image{j} = double( imread( image_path ) );
    female_face = [ female_face female_face_image{j}(:) ];
    female_face_mean = female_face_mean + female_face_image{j};
    j = j + 1;
end
female_face_mean = female_face_mean ./ num_female;
female_face_mean_vec = female_face_mean(:);
face_mean = ( female_face_mean .* num_female + male_face_mean .* num_male ) ./ (num_female + num_male);


% Read training landmarks, calculate mean landmark
j = 1;
male_landmark = [];
landmark_mean = zeros( num_landmarks , 2 );
male_landmark_mean = zeros( num_landmarks , 2 );
for i = 1 : (num_male + 1)
    if(i ~= 58)
        landmark_path = strcat( male_landmark_path, sprintf( '%03d', i-1 ), '_87pt.txt' );
        male_landmark_pos{j} = double( importdata( landmark_path ) );
        male_landmark = [ male_landmark male_landmark_pos{j}(:) ];
        landmark_mean = landmark_mean + male_landmark_pos{j};
        male_landmark_mean = male_landmark_mean + male_landmark_pos{j};
        j = j + 1;
    end
end
male_landmark_mean = male_landmark_mean ./ num_male;
male_landmark_mean_vec = male_landmark_mean(:);

j = 1;
female_landmark = [];
female_landmark_mean = zeros( num_landmarks , 2 );
for i = 1 : (num_female)
    landmark_path = strcat( female_landmark_path, sprintf( '%03d', i-1 ), '_87pt.txt' );
    female_landmark_pos{j} = double( importdata( landmark_path ) );
    female_landmark = [ female_landmark female_landmark_pos{j}(:) ];
    landmark_mean = landmark_mean + female_landmark_pos{j};
    female_landmark_mean = female_landmark_mean + female_landmark_pos{j};
    j = j + 1;
end
female_landmark_mean = female_landmark_mean ./ num_female;
female_landmark_mean_vec = female_landmark_mean(:);

landmark_mean = landmark_mean ./ (num_male + num_female);
landmark_mean_vec = landmark_mean(:);


% Align training faces
male_face_warp = [];
male_face_warp_mean = zeros( image_height, image_width );
for i = 1 : (num_male)
    male_face_warp_image{i} = double( warpImage_new(male_face_image{i}, male_landmark_pos{i}, landmark_mean) );
    male_face_warp = [ male_face_warp male_face_warp_image{i}(:) ];
    male_face_warp_mean = male_face_warp_mean + male_face_warp_image{i};
end
male_face_warp_mean = male_face_warp_mean ./ num_male;


female_face_warp = [];
female_face_warp_mean = zeros( image_height, image_width );
for i = 1 : (num_female)
    female_face_warp_image{i} = double( warpImage_new(female_face_image{i}, female_landmark_pos{i}, landmark_mean) );
    female_face_warp = [ female_face_warp female_face_warp_image{i}(:) ];
    female_face_warp_mean = female_face_warp_mean + female_face_warp_image{i};
end
female_face_warp_mean = female_face_warp_mean ./ num_female;
face_warp_mean = ( female_face_warp_mean .* num_female + male_face_warp_mean .* num_male ) ./ (num_female + num_male);


% Normalize faces
for i = 1 : num_male
    male_face_warp_norm(:, i) = male_face_warp(:, i) - face_warp_mean(:);
    male_landmark_norm(:, i) = male_landmark(:, i) - landmark_mean(:);
end

for i = 1 : num_female
    female_face_warp_norm(:, i) = female_face_warp(:, i) - face_warp_mean(:);
    female_landmark_norm(:, i) = female_landmark(:, i) - landmark_mean(:);
end


% Calculate eigenvalues and eigenvectors
[U, V, S] = pca( [female_face_warp_norm male_face_warp_norm]' );
[UL, VL, SL] = pca( [female_landmark_norm male_landmark_norm]' );


% Calculate Fisher face for key point and appearance, and show them
num_eigenfaces = 20;
w_geometric = myFLD(female_landmark_norm, male_landmark_norm, UL, num_eigenfaces);
w_appearance = myFLD(female_face_warp_norm, male_face_warp_norm, U, num_eigenfaces);

w_appearance_image = reshape( w_appearance, image_height, image_width );

for i = 1 : num_landmarks
    w_geometric_pos(i, 1) = w_geometric(2*i-1);
    w_geometric_pos(i, 2) = w_geometric(2*i);
end


% Calculate projected values and visualize
male_appearance = w_appearance' * male_face_warp_norm;
male_geometric = w_geometric' * male_landmark_norm;

female_appearance = w_appearance' * female_face_warp_norm;
female_geometric = w_geometric' * female_landmark_norm;

figure(1);
hold on
plot(male_appearance, male_geometric, 'b.', 'MarkerSize', 10);
plot(female_appearance, female_geometric, 'r.', 'MarkerSize', 10);
hold off
xlabel('Appearance values');
ylabel('Geometric values');
legend('Male', 'Female');