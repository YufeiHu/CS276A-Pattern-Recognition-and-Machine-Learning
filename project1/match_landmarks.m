% load the faces
for i=1:89
    str=strcat('face_data_new/male_face/',sprintf('face%03d.bmp', i-1) );   %concatenates two strings that form the name of the image
    IO(:,:,i)=double(imread(str));
    class(i) = 1;
end

for i=1:85
    str=strcat('face_data_new/female_face/',sprintf('face%03d.bmp', i-1) );   %concatenates two strings that form the name of the image
    IO(:,:,i+89)=double(imread(str));
    class(i+89) = 2;
end

for i=1:178
    str=strcat('../face_data_new/face/',sprintf('face%03d.bmp', i-1) );   %concatenates two strings that form the name of the image
    F(:,:,i)=double(imread(str));
end

%%
for i=90:174
    for j=1:178
        if sum(IO(:,:,i) - F(:,:,j)) == 0;
            %fprintf('Male %d  = Landmark %d  \n', i-1, j-1)
            fprintf('%d, ', j-1);
        end
    end
end
%%

for i=90:174
    for j=1:178
        if sum(I(:,:,i) - F(:,:,j)) == 0;
            fprintf('Female %d  = Landmark %d  \n', i-90, j-1)
        end
    end
end