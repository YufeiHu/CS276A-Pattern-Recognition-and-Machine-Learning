%% =================== File header ===================
% Title: CS 276A - Pattern Recognition and Machine Learning Project 1
% Subtitle: Problem 5
% Author: Yufei Hu
% Date: 10/21/2017

clc
clear all
close all
set(0, 'defaultfigurecolor', [1 1 1])


%% ========== Find Fisher face ==========
% User definition area
num_train_male = 78;
num_test_male = 10;
num_train_female = 75;
num_test_female = 10;

image_height = 256;
image_width = 256;
num_landmarks = 87;

male_face_path = './face_data_new/male_face/face';
female_face_path = './face_data_new/female_face/face';
male_landmark_path = './face_data_new/male_landmark_87/face';
female_landmark_path = './face_data_new/female_landmark_87/face';


% Read training and images, calculate mean face
j = 1;
male_face_train = [];
male_face_mean = zeros( image_height, image_width );
for i = 1 : (num_train_male + 1)
    if(i ~= 58)
        image_path = strcat( male_face_path, sprintf( '%03d', i-1 ), '.bmp' );
        male_face_train_image{j} = double( imread( image_path ) );
        male_face_train = [ male_face_train male_face_train_image{j}(:) ];
        male_face_mean = male_face_mean + male_face_train_image{j};
        j = j + 1;
    end
end
male_face_mean = male_face_mean ./ num_train_male;
male_face_mean_vec = male_face_mean(:);


j = 1;
female_face_train = [];
female_face_mean = zeros( image_height, image_width );
for i = 1 : (num_train_female)
    image_path = strcat( female_face_path, sprintf( '%03d', i-1 ), '.bmp' );
    female_face_train_image{j} = double( imread( image_path ) );
    female_face_train = [ female_face_train female_face_train_image{j}(:) ];
    female_face_mean = female_face_mean + female_face_train_image{j};
    j = j + 1;
end
female_face_mean = female_face_mean ./ num_train_female;
female_face_mean_vec = female_face_mean(:);
face_mean = ( female_face_mean .* num_train_female + male_face_mean .* num_train_male ) ./ (num_train_male + num_train_female);


% Normalize training faces
for i = 1 : num_train_male
    male_face_train_norm(:, i) = male_face_train(:, i) - face_mean(:);
end

for i = 1 : num_train_female
    female_face_train_norm(:, i) = female_face_train(:, i) - face_mean(:);
end


% Calculate eigenvalues and eigenvectors
[U, V, S] = pca( [female_face_train_norm male_face_train_norm]' );
    

% Calculate Fisher face
num_eigenfaces = 20;
w = myFLD(female_face_train_norm, male_face_train_norm, U, 20);


figure(1)
imshow(reshape(w, image_width, image_height), []);
title('Fisher face', 'fontSize', 20);


% Plot training set
w0 = 0;
class_result_train = [];
score_male_result_train = [];
for i = 1 : num_train_male
    face_norm = male_face_train(:, i) - face_mean(:);
    score_tmp  = w' * face_norm + w0;
    score_male_result_train = [score_male_result_train score_tmp];
    if(score_tmp > 0)
        class_result_train = [class_result_train 1];
    else
        class_result_train = [class_result_train 0];
    end
end

score_female_result_train = [];
for i = 1 : num_train_female
    face_norm = female_face_train(:, i) - face_mean(:);
    score_tmp  = w' * face_norm + w0;
    score_female_result_train = [score_female_result_train score_tmp];
    if(score_tmp > 0)
        class_result_train = [class_result_train 1];
    else
        class_result_train = [class_result_train 0];
    end
end


figure(2)
hold on
x_axis = linspace(1, num_train_male, num_train_male);
plot(x_axis, score_male_result_train, 'b.', 'MarkerSize', 20);
x_axis = linspace(1, num_train_female, num_train_female);
plot(x_axis, score_female_result_train, 'r.', 'MarkerSize', 20);
x_axis = linspace(1, max(num_train_male, num_train_female), max(num_train_male, num_train_female));
y_axis = w0 .* ones(1, max(num_train_male, num_train_female));
plot(x_axis, y_axis, 'k--', 'LineWidth', 3);
hold off
legend('Male', 'Female', 'Classification Boundary');
ylabel('Scores');
title('Training data classification results');


% Test
j = 1;
male_face_test = [];
male_face_test_mean = zeros( image_height, image_width );
for i = (num_train_male + 2) : (num_train_male + num_test_male + 1)
    image_path = strcat( male_face_path, sprintf( '%03d', i-1 ), '.bmp' );
    male_face_test_image{j} = double( imread( image_path ) );
    male_face_test = [ male_face_test male_face_test_image{j}(:) ];
    male_face_test_mean = male_face_test_mean + male_face_test_image{j};
    j = j + 1;
end
male_face_test_mean = male_face_test_mean ./ num_test_male;

j = 1;
female_face_test = [];
female_face_test_mean = zeros( image_height, image_width );
for i = (num_train_female + 1) : (num_train_female + num_test_female)
    image_path = strcat( female_face_path, sprintf( '%03d', i-1 ), '.bmp' );
    female_face_test_image{j} = double( imread( image_path ) );
    female_face_test = [ female_face_test female_face_test_image{j}(:) ];
    female_face_test_mean = female_face_test_mean + female_face_test_image{j};
    j = j + 1;
end
female_face_test_mean = female_face_test_mean ./ num_test_female;


class_result = [];
score_male_result = [];
for i = 1 : num_test_male
    face_norm = male_face_test(:, i) - face_mean(:);
    score_tmp = w' * face_norm + w0;
    score_male_result = [score_male_result score_tmp];
    if(score_tmp > 0)
        class_result = [class_result 1];
    else
        class_result = [class_result 0];
    end
end

score_female_result = [];
for i = 1 : num_test_female
    face_norm = female_face_test(:, i) - face_mean(:);
    score_tmp = w' * face_norm + w0;
    score_female_result = [score_female_result score_tmp];
    if(score_tmp > 0)
        class_result = [class_result 1];
    else
        class_result = [class_result 0];
    end
end


figure(3)
hold on
x_axis = linspace(1, num_test_male, num_test_male);
plot(x_axis, score_male_result, 'b.', 'MarkerSize', 20);
x_axis = linspace(1, num_test_female, num_test_female);
plot(x_axis, score_female_result, 'r.', 'MarkerSize', 20);
x_axis = linspace(1, max(num_test_male, num_test_female), max(num_test_male, num_test_female));
y_axis = w0 .* ones(1, max(num_test_male, num_test_female));
plot(x_axis, y_axis, 'k--', 'LineWidth', 3);
hold off
legend('Male', 'Female', 'Classification Boundary');
ylabel('Scores');
title('Testing data classification results');


figure(4)
hold on
histogram([score_female_result_train score_female_result], 12, 'FaceColor', 'r');
histogram([score_male_result_train score_male_result], 12, 'FaceColor', 'b');
hold off
legend('Female', 'Male');
xlabel('Scores');
ylabel('Number');

