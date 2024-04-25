clear all
close all

rgb = imread("DJI_0045.jpg");
In = im2gray(rgb);
%I(I > 140) = 0;
%I(:,1:1200) = 0;
In = rgb2gray(rgb);
II = rgb2gray(rgb);
In(:,1:680) = 0;
In(:,2760:4032) = 0;
%In(1:550,:) = 0;
%In(2450:3024,:) = 0;
imshow(In)


%A_14(1:12,1:1420) = 0;

% AAA=A_36;
% AAA(:,2450:4032) = 0;
% AAA(1:1630,:) = 0;
% AAA(:,1:2210) = 0;
% AAA(2115:3024,:) = 0;
%A_13(:,1755:4032) = 0;

text(732,501,"Image courtesy of Corel(R)",...
     "FontSize",7,"HorizontalAlignment","right")

figure
gmag = imgradient(In);
imshow(gmag,[])
title("Gradient Magnitude")

figure
L = watershed(gmag);
Lrgb = label2rgb(L);
imshow(Lrgb)
title("Watershed Transform of Gradient Magnitude")

figure
se = strel("disk",20);
Io = imopen(In,se);
imshow(Io)
title("Opening")

figure
Ie = imerode(In,se);
Iobr = imreconstruct(Ie,In);
imshow(Iobr)
title("Opening-by-Reconstruction")

figure
Ioc = imclose(Io,se);
imshow(Ioc)
title("Opening-Closing")

figure
Iobrd = imdilate(Iobr,se);
Iobrcbr = imreconstruct(imcomplement(Iobrd),imcomplement(Iobr));
Iobrcbr = imcomplement(Iobrcbr);
imshow(Iobrcbr)
title("Opening-Closing by Reconstruction")

figure
fgm = imregionalmax(Iobrcbr);
imshow(fgm)

figure
I2 = labeloverlay(In,fgm);
imshow(I2)
title("Regional Maxima Superimposed on Original Image")

se2 = strel(ones(5,5));
fgm2 = imclose(fgm,se2);
fgm3 = imerode(fgm2,se2);
fgm4 = bwareaopen(fgm3,20);
I3 = labeloverlay(In,fgm4);
% imshow(I3)
% title("Modified Regional Maxima Superimposed on Original Image")
% title("Regional Maxima of Opening-Closing by Reconstruction")

bw = imbinarize(Iobrcbr);
% imshow(bw)
% title("Thresholded Opening-Closing by Reconstruction")

D = bwdist(bw);
DL = watershed(D);
bgm = DL == 0;
% imshow(bgm)
% title("Watershed Ridge Lines")

gmag2 = imimposemin(gmag, bgm | fgm4);
L = watershed(gmag2);
%L(:,1:540) = 0;
labels = imdilate(L==0,ones(3,3)) + 2*bgm + 3*fgm4;
I4 = labeloverlay(In,labels);
% imshow(I4)
% title("Markers and Object Boundaries Superimposed on Original Image")

figure
Lrgb = label2rgb(L,"jet","w","shuffle");
imshow(Lrgb)
title("Colored Watershed Label Matrix")

% Get the size of the image
[rows, cols] = size(bw);

% figure
% imshow(In)
% hold on
% himage = imshow(Lrgb);
% himage.AlphaData = 0.3;
% title("Colored Labels Superimposed Transparently on Original Image")

%Lrgb(Lrgb>4)=2;
% Lrgb(Lrgb ~= 3) = 0;
% Lrgb(Lrgb == 3) = 5;
% figure
LL = im2gray(Lrgb);
% imshow(LL);
% LL(LL == 196) = 0;
% AA(AA ~= 0) = 1;
% Count the number of unique labels in the segmented image
num_classes = numel(unique(L)) - 1; % Exclude background label

%disp(['Number of classes or segments in the segmented image: ', num2str(num_classes)]);

% Assuming your matrix is named 'A'
% Find the indices of the non-zero elements equal to 2
[row, col] = find(LL == 255);

% Initialize a list to store neighboring values
neighbors = [];

% Check the neighbors of each non-zero element equal to 2
for i = 1:length(row)
    % Check top neighbor
    if row(i) > 1
        neighbors = [neighbors LL(row(i)-1, col(i))];
    end
    % Check bottom neighbor
    if row(i) < size(LL, 1)
        neighbors = [neighbors LL(row(i)+1, col(i))];
    end
    % Check left neighbor
    if col(i) > 1
        neighbors = [neighbors LL(row(i), col(i)-1)];
    end
    % Check right neighbor
    if col(i) < size(LL, 2)
        neighbors = [neighbors LL(row(i), col(i)+1)];
    end
end

% Find unique values in the list
uniqueNeighbors = unique(neighbors);

% Remove the value 2 itself if it's in the list
uniqueNeighbors(uniqueNeighbors == 255) = [];

% Display the unique neighboring values
disp('Unique values adjacent to 255:');
disp(uniqueNeighbors);

% Assuming uniqueNeighbors contains the unique neighboring values
num_unique = length(uniqueNeighbors);
matrices = cell(1, num_unique);

% Iterate over each unique neighbor value
for i = 1:num_unique
    % Create a matrix filled with the neighboring value
    current_neighbor = uniqueNeighbors(i);
    matrix = LL;
    % Set elements not equal to the current neighbor value to 0
    matrix(matrix ~= current_neighbor) = 0;
    matrix(matrix == current_neighbor) = 255;
    % Store the matrix in the cell array
    matrices{i} = matrix;
    % Optionally, you can also display each matrix
    
end
AA = zeros(size(LL));
for i = 1:length(matrices)
    eval(['A_' num2str(i) ' = matrices{:, i};']);
    
end

% AA = A_1 + A_2 + A_3 + A_4 + A_5 + A_6 + A_7 + A_8 + A_9 + A_10 + A_11 + A_12 + A_13 + A_14 + A_15 + A_16 + A_17 + A_18 + A_19 + A_20 + A_21 + A_22 + A_23 + A_24 + A_25 + A_26 + A_27 + A_28 + A_29 + A_30 + A_31 + A_32 + A_33 + A_34 + A_35+ ...
%      + A_36 + A_37 + A_38 + A_39 + A_40 + A_41 + A_42 + A_43 + A_44 + A_45+ A_46 + A_47 + A_48 + A_49 + A_50 + A_51;
% 
%AA = A_1 + A_4 + A_5 + A_6 + A_7 + A_8 + A_10 + A_11 + A_12 + A_13 + A_15 + A_17 + A_18 + A_19 + A_20 + A_21 + A_22 + A_23 + A_24 + A_25 + A_26 + A_27 + A_28 + A_29 + A_30 + A_31 + A_32 + A_33 + A_34 + A_35+ ...
%     + A_36 + A_37 + A_38 + A_39 + A_40 + A_41 + A_42 + A_43 + A_44 + A_45+ A_46 + A_47 + A_48 + A_49 + A_50 + A_51;


%AA = A_3 + A_4 + A_5 + A_8 + A_9 + A_10 + A_11 + A_12 + A_13 + A_14 + A_15 + A_16 + A_17 + A_18 + A_19 + A_20 + A_21 + A_22 + A_24 + A_25 + A_26 + A_27 + A_28 + A_29 + A_30 + A_33 + A_34+A_35+A_36;
%AA = A_1 + A_2 + ... + A_n;

%AA = A_2 + A_3 + A_4 + A_5 + A_6 + A_7 + A_8 + A_9 + A_10 + A_11 + A_12 + A_13 +A_15 + A_16 + A_17 + A_18 + A_19 + A_20 + A_22 + A_23 + A_24 + A_25 + A_27 + A_28 + A_29 + A_30 + A_31 + A_32+ A_34 + A_35+ ...
%     + A_36 + A_37 + A_38 + A_40 + A_42 + A_43 + A_44 + A_46 + A_47 + A_48 + A_49 + A_50 + A_51+ A_53 + A_55+ A_56 + A_57 + A_58 +A_59+A_60;

%AA = A_3 + A_4 + A_5 + A_8 + A_9 + A_10 + A_11 + A_12+A_14;
% AA =  A_2 + A_3 + A_4 + A_5 + A_6 + A_7  + A_11 +  A_14 + A_15 +...
%      A_16 + A_17 + A_18 + A_19 + A_20  + A_23  + A_25 + A_26 + A_28 + ...
%      A_30 + A_31+  A_32 + A_34 + A_35+ A_36+ A_42+ A_43 + A_44+ A_45 + A_46 + A_47 + A_48 + A_49 +  A_51+  + A_53;

%AA = A_1 + A_2 + A_3 + A_4 + A_5+  A_6 + A_7+  A_8 + A_9 + A_10 + A_11 +  A_12;+...
% A_13+ A_14 + A_15 + A_16 + A_17+A_18 + A_19 + A_20 + A_21 + A_22 + A_23+...
% A_24 + A_25+ A_26+A_27+A_28 + A_29 + A_30 + A_31 + A_32 + A_33+ A_34 + A_35+ A_36 + A_37+A_38 + A_39+...
% A_40 + A_41 + A_42+ A_43;+ A_44 + A_45; + A_46 + A_47+A_48 + A_49+A_50+ A_51 + A_52 + A_53+ A_54 + A_55; + A_56 + A_57;
% 
% unique_values = unique(CC);
% disp('Unique values in the matrix:');
% disp(unique_values);

% numel(matrices)
for i = 1:numel(matrices)
    figure;
    imshow(In);
    hold on
    himage =imshow(matrices{i});
    himage.AlphaData = 0.3; 
end

% for i = 1:43
%     figure;
%     imshow(In);
%     str = ['A_',num2str(i)];
%     eval(['dummy =',str,';']);
%     hold on
%     himage =imshow(dummy);
%     himage.AlphaData = 0.3; 
% end


% 
% figure
% imshow(In)
% hold on
% himage = imshow(AA);
% himage.AlphaData = 0.3;
% title("Colored Labels Superimposed Transparently on Original Image")


% figure
% imagesc(A_6);
% 
% % Set the axes limits to cover the range from 1 to 4000
% xlim([1 4000]);
% ylim([1 4000]);