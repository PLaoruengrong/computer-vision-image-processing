clear all
close all
%%%% Read in a test image with a door %%%%%

% First test image 
image = imread('./01 - R2441 - i.JPG');
[M N C] = size(image);
figure(1),imshow(image,'Border','tight');

% Second test image
%image = imread('./01 - R2442 - i.JPG');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Task 1: Image Preprocessing - Contrast Adjustment, Noise Reduction      %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%% Conversion to Gray Value Image %%%%

imgGray = rgb2gray(image);

%%%% Adjust Contrast %%%%

gamma=0.5;
imgComp=imadjust(imgGray,[],[],gamma);

%%%% Noise reduction via binomial low-pass filtering %%%% 

% Apply binomial low-pass filter for noise reduction
binom = [1 4 6 4 1]/16;
img = imfilter(imfilter(imgComp,binom,'replicate'),binom','replicate');
% img = imfilter(imfilter(imgComp,binom,'symmetric'),binom','symmetric');
% img = imfilter(imfilter(imgComp,binom,'circular'),binom','circular');
 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Task 2: Feature Extraction - Contours                                   %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%% Optimized Sobel x %%%%

Sobel=[-3 0 3;-10 0 10; -3 0 3]/32;
[U,S,V]=svd(Sobel);
S(2,2)
U(:,1)
V(:,1)
u1 = U(:,1) * sqrt(S(1,1));
v1 = V(:,1) * sqrt(S(1,1));
img=double(img);
img_sobel_x = imfilter(imfilter(img, u1, 'replicate'), v1', 'replicate');

%%%% Edge Strength %%%%

% Compute edge strength for x-direction

img_edge_strength_x = abs(img_sobel_x);

% Negative x-Gradients (bright to dark)  %

% Define edge strength threshold %

threshX = 3*mean(img_edge_strength_x(:));

% Segment strong negative x-Gradients %

binNegX = (img_sobel_x < -threshX);

% Noise reduction %

se_noise = ones(2,2); 
binNegXRedNoi = imopen(binNegX, se_noise);

% Dilate for 1 pixel %

se_dilate = strel('diamond', 1);
img_neg_edge_x = imdilate(binNegXRedNoi, se_dilate);

% Positive x-Gradients (dark to bright) %

% Segment strong positive x-Gradients %

binPosX = (img_sobel_x > threshX);

% Noise reduction %

binPosXRedNoi = imopen(binPosX, se_noise);

% Dilate for 1 pixel %

img_pos_edge_x = imdilate(binPosXRedNoi, se_dilate);

% Optimized Sobel y %

dIy = imfilter(imfilter(img, v1, 'replicate'), u1', 'replicate');

% Negative y-Gradients (bright to dark)  %

% Define y-edge strength threshold %

magGradY = abs(dIy);
threshY = 3*mean(magGradY(:));

% Segment strong negative y-Gradients %

binNegY = (dIy < -threshY);

% Noise reduction %

binNegYRedNoi = imopen(binNegY, se_noise);

% Dilate for 1 pixel %

img_neg_edge_y = imdilate(binNegYRedNoi, se_dilate);

% Positive y-Gradients (dark to bright) %

% Segment strong positive y-Gradients %

binPosY = (dIy > threshY);

% Noise reduction %

binPosYRedNoi = imopen(binPosY, se_noise);

% Dilate for 1 pixel %

img_pos_edge_y = imdilate(binPosYRedNoi, se_dilate);


%%%% Visualize Results %%%%
figure(2),imagesc(img_sobel_x); colorbar; colormap hsv; axis off;
figure(3),imagesc(log(1+img_edge_strength_x)); colorbar; axis off;
figure(4),imagesc(img_neg_edge_x); colorbar; colormap gray; axis off;
figure(5),imagesc(img_pos_edge_x); colorbar; colormap gray; axis off;
figure(6),imagesc(img_neg_edge_y); colorbar; colormap gray; axis off;
figure(7),imagesc(img_pos_edge_y); colorbar; colormap gray; axis off;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Task 3: Segmentation of Door Gap Contour                                %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Extract Vertical Edge candidates for % symmetrical light-dark-light
% transition

shiftedVerGreen = [binPosX(:, 4+1:end), zeros(size(binPosX, 1), 4)]; % shift green line 4 pixels to the left

% Design a symmetric filter % by use pointwise logical operation:
% & for fixed red, shifted green and dark line are aligned
% | for combine the vertical and horizontal gaps

% Combine knowledge on gray value range and bright-to-dark gradients 

tic
imgGrayMean=round(sum(sum(img))/(size(img,1)*size(img,2)))
toc
imgBinaer=(img < imgGrayMean);
verCandidatesL = binNegX & shiftedVerGreen & imgBinaer;

% Fatten lines to improve detection 

candidates_x = imdilate(verCandidatesL , ones(5,1));

% Extract Horizontal Edge candidates 

shiftedHorGreen = [binPosY(4+1:end, :); zeros(4, size(binPosY, 2))]; % shift green line 4 pixels to upward

% Combine knowledge on absolute gray value and bright-to-dark-gradient

horCandidatesL = binNegY & shiftedHorGreen & imgBinaer;

% Fatten lines to improve detection 

candidates_y = imdilate(horCandidatesL , ones(1,5));

doorGap = candidates_x | candidates_y;
doorGapRedNoi = imopen(doorGap, ones(3,3));
img_sym_x = imdilate(doorGapRedNoi, ones(3,3));


%%%% Visualize Results %%%%
figure(8),imagesc(max(max(img_sym_x))-abs(img_sym_x));colorbar; colormap gray;axis off;
figure(9),imagesc(1-candidates_x);colorbar; colormap gray;axis off;
figure(10),imagesc(1-candidates_y);colorbar; colormap gray;axis off

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Task 4: Measuring Lines of the Door Gap                                 %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% Compute Hough space for vertical lines

[M, N] = size(img);
HVer = htLine(candidates_x);

%%%% Find vertical Hough Lines %%%%

HVer = double(HVer);
l_x = zeros(3, 2); % Hessian normal form: [l1; l2; l3] -> l1*x + l2*y + l3 = 0

for k = 1:2
    [max_val, idx] = max(HVer(:));
    [row, col] = ind2sub(size(HVer), idx);
    phi=col*pi/180;
 
% Transform two best candidates to Hesse Normal form
    l_x(:, k) = [sin(phi); cos(phi); -row]; % l1*x + l2*y + l3 = x*cos(phi) + y*sin(phi) - d = 0
    rows = max(1, row-20):min(size(HVer,1), row+20);
    cols = max(1, col-20):min(size(HVer,2), col+20);
    HVer(rows, cols) = 0;
end

% Compute Hough space for horizontal lines

HHor = htLine(candidates_y); 

%%%% Find horizontal Hough Lines %%%%

HHor = double(HHor);
l_y = zeros(3, 2);

for k = 1:2
    [max_val, idx] = max(HHor(:));
    [row, col] = ind2sub(size(HHor), idx);
    phi = col * pi/180;
    
% Transform two best candidates to Hesse Normal form

    l_y(:, k) = [sin(phi); cos(phi); -row]; % l1*x + l2*y + l3 = x*cos(phi) + y*sin(phi) - d = 0
    rows = max(1, row-20):min(size(HHor,1), row+20);
    cols = max(1, col-20):min(size(HHor,2), col+20);
    HHor(rows, cols) = 0;
end


%%%% Visualize Results %%%%
figure(11),imshow(image,'Border','tight');hold on;
for k=1:2,
    plot([round(-(l_x(2,k)+l_x(3,k))/l_x(1,k)); round(-(M*l_x(2,k)+l_x(3,k))/l_x(1,k));],...
    [1,M],...
    'LineWidth',1,'Color','green');hold on;
end
for k=1:2,
plot([1,N],...
    [round(-(l_y(1,k)+l_y(3,k))/l_y(2,k)); round(-(N*l_y(1,k)+l_y(3,k))/l_y(2,k));],... 
    'LineWidth',1,'Color','green');hold on;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Task 5: Measure and Classify Corner Points                              %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Find left and right door gap lines %

[~, sortIdxX] = sort(abs(l_x(3, :))); % sort Hessian normal form of stright line based on -d (distance)
l_x_left  = l_x(:, sortIdxX(1)); % smaller l3 is left side
l_x_right = l_x(:, sortIdxX(2));

% Find up and down door gap lines %

[~, sortIdxY] = sort(abs(l_y(3, :)));
l_y_up  = l_y(:, sortIdxY(1)); 
l_y_down = l_y(:, sortIdxY(2));

% Find corner points %

% Left-Up (lu)
p_lu = cross(l_x_left, l_y_up); % cross product of 2 lines gives coordinate
lu = p_lu(1:2) / p_lu(3); % normalize to get Euclidean [x; y; 1]

% Right-Up (ru)
p_ru = cross(l_x_right, l_y_up);
ru = p_ru(1:2) / p_ru(3);

% Left-Bottom (lb)
p_lb = cross(l_x_left, l_y_down);
lb = p_lb(1:2) / p_lb(3);

% Right-Bottom (rb)
p_rb = cross(l_x_right, l_y_down);
rb = p_rb(1:2) / p_rb(3);

%%%% Visualize Results %%%%
figure(12),imshow(image,'Border','tight');hold on;
plot([round(-(l_x_left(2)+l_x_left(3))/l_x_left(1)); round(-(M*l_x_left(2)+l_x_left(3))/l_x_left(1));],...
    [1,M],...
    'LineWidth',1,'Color','red');hold on;
plot([round(-(l_x_right(2)+l_x_right(3))/l_x_right(1)); round(-(M*l_x_right(2)+l_x_right(3))/l_x_right(1));],...
    [1,M],...
    'LineWidth',1,'Color','green');hold on;
plot([1,N],...
    [round(-(l_y_up(1)+l_y_up(3))/l_y_up(2)); round(-(N*l_y_up(1)+l_y_up(3))/l_y_up(2));],... 
    'LineWidth',1,'Color','red');hold on;
plot([1,N],...
    [round(-(l_y_down(1)+l_y_down(3))/l_y_down(2)); round(-(N*l_y_down(1)+l_y_down(3))/l_y_down(2));],... 
    'LineWidth',1,'Color','green');hold on;
plot(lb(1),lb(2),'ro','MarkerFaceColor','r');hold on;
plot(lu(1),lu(2),'go','MarkerFaceColor','g');hold on;
plot(ru(1),ru(2),'bo','MarkerFaceColor','b');hold on;
plot(rb(1),rb(2),'ko','MarkerFaceColor','k');hold on;
