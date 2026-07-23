%% Generate coded marker list
% This section generates a set of unique cyclic binary codes and saves them
% to codebook.mat. The codebook is used by both the loss function (training)
% and the NMS decoder (validation/inference) to map 15-bit binary predictions
% to class IDs.
%
% Algorithm:
%   1. Generate all (2^15 - 2) possible 15-bit binary codes
%   2. For each code, compute the minimum value across all 15 cyclic shifts
%      (this is the "cycle-to-minimum" operation for rotation invariance)
%   3. Remove duplicate minimum values, keeping only unique codes
%   4. Build a lookup table mapping any 15-bit integer to its 1-indexed code ID
%   5. Save as codebook.mat for use by the Python training pipeline
%
% The resulting codebook has 2190 unique cyclic codes out of 32766 possible.

% clear;
codeSize = 15; % number of bits in the binary code
size_ = 55*5;  % image pixel size for a normal-sized marker
size_half = (size_ + 1) / 2 - 1;

% Generate all possible 15-bit binary code sequences (excluding 0 and all-1s)
numCodes = 2^codeSize - 2;
oriCodeList = arrayfun(@(x) dec2bin(x, codeSize), 1:numCodes, 'UniformOutput', false);

% Compute the minimum value across all cyclic shifts for each code
% This is the core of cycle-invariant coding: each rotation of the same
% marker maps to the same minimum value, providing rotational invariance.
minValueList = zeros(1, numCodes);
for i = 1:numCodes
    tempCode = (oriCodeList{i});
    currValue = base2dec(tempCode, 2);
    for j = 1:codeSize-1
        tempCode = circshift(tempCode, 1);
        tempValue = base2dec(tempCode, 2);
        if tempValue < currValue
            currValue = tempValue;
        end
    end
    minValueList(i) = currValue;
end

% Remove duplicate minimum values to obtain the unique cyclic code set
% Each unique minimum value corresponds to one rotation-invariant code class.
codeList = [];
codeValueList = [];
numDiff = 0;
isExist = 0;
for i=1:2^codeSize-2
    isExist = 0;
    for j=1:size(codeValueList,2)
        if minValueList(i) == codeValueList(j)
            isExist = 1;
            break;
        end
    end
    if isExist == 0
        numDiff = numDiff+1;
        codeValueList(numDiff) = minValueList(i);
        codeList{numDiff} = oriCodeList{i};
    end
end

% Build the codebook lookup tables
% codeValueList_map: For each possible 15-bit integer (used as index), 
%   stores its corresponding 1-indexed code ID if it is a cyclic minimum; 
%   otherwise, stores 0 (meaning the index is not in the set of cyclic minima).
% minValueList_int: For each possible 15-bit integer (used as index), 
%   stores the cyclic minimum value (1-indexed code ID) that this index maps to.
% codeValueList_int: For each 1-indexed code ID (used as index, range 1–2190), 
%   stores its corresponding cyclic minimum value (15-bit integer).
length_max = max(codeValueList);
codeValueList_map = zeros(1, length_max);
codeValueList_map(codeValueList) = 1:2190;
codeValueList_map_int = int32(codeValueList_map);    % convert to signed 32-bit for MATLAB save
minValueList_int  = int32(minValueList);             % convert to signed 32-bit
codeValueList_int = int32(codeValueList);            % convert to signed 32-bit
save('codebook.mat', 'codeValueList_map_int', 'minValueList_int', 'codeValueList_int');
return


%% Draw coded markers and save as PNG images
% This section renders each unique cyclic code as a visual marker image.
% The marker uses a concentric ring + radial sector design:
%   - Outer solid circle (white background for the code)
%   - 15 radial sectors at the outer ring, each either white (bit=1) or black (bit=0)
%   - Inner solid ring (always black, provides a clear boundary)
%   - Center solid dot (always white, provides a reference point)

% Image and encoding parameter setup
size_high_resolution = size_;
[X, Y] = meshgrid(1:size_high_resolution, 1:size_high_resolution);  % pixel grid
cen_x = (size_high_resolution+1)/2;
cen_y = cen_x;
size_half_resolution = cen_x-1;
distance_from_center = sqrt((X - cen_x).^2 + (Y - cen_y).^2);  % distance of each pixel from center
ang_from_center = atan2((X - cen_x), (Y - cen_y));              % angle of each pixel from center

% Geometric scaling for marker elements
point_size_time = 2*size_half_resolution/64;  % circle scaling factor
r0 = point_size_time*10/2;     % base center circle radius = 5 px
lb = r0/2.5*27;                % base side length = 54 px
rin = r0/2.5*6.5;              % base inner ring radius = 13 px
rout = r0/2.5*10;              % base outer ring radius = 20 px

% Create the angular mask for the 15 radial sectors (one per code bit)
ang_mask = zeros(size_high_resolution, size_high_resolution);  % angle mask
angle_step = 2*pi/codeSize;                % each sector spans 360/15 degrees
sector_angle_step = -pi:angle_step:pi;     % sector boundaries
for i=1:codeSize
    ang_mask(ang_from_center > sector_angle_step(i)) = i;
end

% Create output directories for marker images
marker_path = './coded_marker/';
tra_path = [marker_path, '/'];
if exist(marker_path, 'dir') ~= 7 || exist(tra_path, 'dir') ~= 7
    mkdir(marker_path);
    mkdir(tra_path);
end

% Render each unique code as a marker image
for i=1:length(codeList)
    img = zeros(size_high_resolution, size_high_resolution);

    % Solid outer ring: all pixels within rout are initially white
    img(distance_from_center <= rout) = 1;

    % Encode the binary code: sectors with bit=0 are set to black
    zeros_idx = find(codeList{i} == '0');
    for j=1:length(zeros_idx)
        img(ang_mask == zeros_idx(j)) = 0;
    end

    % Inner ring: always black (provides separation between code ring and center dot)
    img(distance_from_center <= rin) = 0;

    % Center dot: always white (reference point for detection)
    img(distance_from_center <= r0) = 1;

    % Save as PNG
    imwrite(uint8(img*255), [tra_path num2str(i) '.png']);
end

return


%% Parameters for synthetic dataset generation (configurable)
% COCO dataset background images
% This section composites coded markers onto COCO background images with
% random perspective transforms to create a realistic training/validation set.

mode_all = {'train', 'val'};           % dataset split: 'train' or 'val'
pic_train = 2000;                       % number of training images to generate
num_classes = 200;                   % number of unique marker classes to use

pic_val = ceil(pic_train/10);           % validation images = 10% of training
pic_total = [pic_train, pic_val];
pic_begin = [0, pic_train];

% ------------------- Composition parameters (directly defined) -------------------
w = 1280;      % output image width (pixels)
h = 1024;      % output image height (pixels)
p_min = 30;    % minimum marker projection size (pixels)
p_max = 180;   % maximum marker projection size (pixels)
d = 5500;      % focal distance (pixels)

% Rotation angle ranges (degrees)
ax_range = [-75, 75];    % polar angle (tilt, out-of-plane rotation)
az_range = [0, 360];     % azimuthal angle (in-plane rotation)

marker_folder = './dataset/';

for train_val = 1:2
    % Create directory names for this split
    data_name = [mode_all{train_val}, '_moni_' num2str(pic_total(train_val)) '_' num2str(num_classes) '/'];

    % Load COCO background images with augmentation
    imds = imageDatastore(['../datasets/coco/images/' mode_all{train_val} '2017']);
    augmenter = imageDataAugmenter(...
        'RandRotation',       [-10, 10], ...
        'RandXTranslation',   [-20, 20], ...
        'RandYTranslation',   [-20, 20], ...
        'RandXScale',         [0.8, 1.2], ...
        'RandYScale',         [0.8, 1.2], ...
        'RandXReflection',    true, ...
        'RandYReflection',    true);
    augimds = augmentedImageDatastore([1024, 1280], imds, ...
        'DataAugmentation', augmenter, 'ColorPreprocessing', 'rgb2gray');
    augimds.MiniBatchSize = 1;

    % Pre-load all augmented background images
    coco_data = cell(pic_total(train_val), 1);
    reset(augimds);
    for i = 1:pic_total(train_val)
        coco_data{i} = read(augimds);
    end

    % Create output directories for images and labels
    im_path = [marker_folder, 'images/', data_name];
    lb_path = [marker_folder, 'labels/', data_name];
    if exist(im_path, 'dir') ~= 7
        mkdir(im_path);
        mkdir(lb_path);
    end

    patch_half = 0.5;
    % Camera pixel coordinates for the marker corners (used in homography)
    patch_roi_uc = [-patch_half, -patch_half, 1;  patch_half, -patch_half, 1;...
        patch_half, patch_half, 1; -patch_half, patch_half, 1; 0, 0, 1];  % camera pixel coordinates

    parfor pic_num = 1:pic_total(train_val)
        bg_image = double(coco_data{pic_num}.input{1,1});

        % Seed random generator for reproducibility
        rng((pic_begin(train_val)+pic_num) * 1000);
        para_all = rand(100, 9);    % random parameters for up to 100 markers per image
        mask_all = zeros(h, w);       % occupancy mask to prevent marker overlap

        fileID = fopen([lb_path, num2str(pic_num) '.txt'], 'w');
        for i = 1:100
            % ---- Template loading and preprocessing ----
            % Randomly select a marker template
            pic_idx = floor(num_classes * para_all(i,1)) + 1;
            template = imread(['./coded_marker/' num2str(pic_idx) '.png']);
            template = double(template);                    % convert to double for computation
            template(template == 0) = para_all(i,2)*10;     % black areas have some reflectivity
            template = template * (para_all(i,3)*0.5 + 0.5); % white exposure adjustment
            template = uint8(template);                     % convert back to uint8

            % Template side length (must be square)
            m = size(template, 1);
            assert(size(template,2) == m, 'Template must be square');

            % ---- Random rotation angles ----
            % gamma: in-plane rotation (azimuth), alpha: out-of-plane tilt (polar)
            gamma = az_range(1) + para_all(i,4)*(az_range(2)-az_range(1));
            alpha = ax_range(1) + para_all(i,5)*(ax_range(2)-ax_range(1));

            % ---- Rotation matrices ----
            Rz = [cosd(gamma), -sind(gamma), 0;
                  sind(gamma),  cosd(gamma), 0;
                  0,            0,           1];
            Rx = [1, 0,            0;
                  0, cosd(alpha), -sind(alpha);
                  0, sind(alpha),  cosd(alpha)];
            R = Rx * Rz;

            % ---- Perspective projection via homography ----
            % Compute the depth offset (Z translation) for desired marker size
            t_z_max = d * (m/p_min - 1);
            t_z_min = d * (m/p_max - 1);
            t_z = t_z_min + para_all(i,6)*(t_z_max - t_z_min);
            scale_t = d / (d + t_z);
            l_proj = m * scale_t;    % projected marker side length (pixels)

            % Random (X, Y) translation to keep the marker within the canvas
            tx_min = (l_proj/2 - w/2) * (d + t_z)/d;
            tx_max = (w - l_proj/2 - w/2) * (d + t_z)/d;
            ty_min = (l_proj/2 - h/2) * (d + t_z)/d;
            ty_max = (h - l_proj/2 - h/2) * (d + t_z)/d;
            if tx_min >= tx_max, tx_min = tx_max; end
            if ty_min >= ty_max, ty_min = ty_max; end
            t_x = tx_min + para_all(i,7)*(tx_max - tx_min);
            t_y = ty_min + para_all(i,8)*(ty_max - ty_min);

            % Construct the full homography matrix H (3x3)
            % Maps from marker template coordinates to image coordinates
            H = [d*R(1,1)+(w+1)/2*R(3,1), d*R(1,2)+(w+1)/2*R(3,2), d*t_x+(w+1)/2*(d+t_z);
                 d*R(2,1)+(h+1)/2*R(3,1), d*R(2,2)+(h+1)/2*R(3,2), d*t_y+(h+1)/2*(d+t_z);
                 R(3,1),                   R(3,2),                   d+t_z] * ...
                [1, 0, -(m+1)/2;
                 0, 1, -(m+1)/2;
                 0, 0,  1];
            H = H / H(3,3);    % normalize so that H(3,3) = 1

            % ---- Step 1: Compute projected marker corners and center ----
            [h_pad, w_pad] = size(template);
            corners = [1, 1, 1; 1, h_pad, 1; w_pad, h_pad, 1; w_pad, 1, 1; (m+1)/2, (m+1)/2, 1];
            corners_t = H * corners';
            corners_t = corners_t ./ corners_t(3,:);
            xywh = [corners_t(1,5), corners_t(2,5), l_proj, l_proj];  % (xc, yc, w, h)

            % ---- Step 2: Create quadrilateral mask of pixels to warp ----
            mask = poly2mask(corners_t(1,1:4), corners_t(2,1:4), h, w);
            [ydst, xdst] = find(mask);

            % ---- Step 3: Compute corresponding source pixels in the template ----
            Hinv = inv(H);
            refX = xywh(1);
            refY = xywh(2);
            corners = [ ...
                refX-0.5, refX+0.5, refX+0.5, refX-0.5;
                refY-0.5, refY-0.5, refY+0.5, refY+0.5;
                1,        1,        1,        1];
            src = Hinv * corners;
            src(1,:) = src(1,:) ./ src(3,:);
            src(2,:) = src(2,:) ./ src(3,:);
            maskSrc = poly2mask(src(1,:), src(2,:), h, w);
            [ys, xs] = find(maskSrc);
            dx = xs - (m+1)/2;
            dy = ys - (m+1)/2;

            % ---- Step 4: Inverse warp: average template pixels for each output pixel ----
            J = zeros(h, w);
            for k = 1:numel(xdst)
                xd = xdst(k);
                yd = ydst(k);
                p = Hinv * [xd; yd; 1];
                p = p(1:2) / p(3);
                xq = round(p(1) + dx);
                yq = round(p(2) + dy);
                valid = xq >= 1 & xq <= m & yq >= 1 & yq <= m;
                if any(valid)
                    idx = sub2ind([m, m], yq(valid), xq(valid));
                    J(yd, xd) = mean(template(idx));
                end
            end
            warped = uint8(round(J));

            % Apply Gaussian blur to simulate defocus/imperfect imaging
            warped = imgaussfilt(warped, 1*para_all(i,9));

            % ---- Check for overlap with existing markers ----
            if sum(mask_all(mask)) > 1
                % Skip this marker if it overlaps with previously placed markers
                continue;
            else
                % Composite the marker onto the background image
                bg_image(mask) = warped(mask);
                mask_all(mask) = 1;

                % Write the YOLO-format label: class_id xc yc w h
                % Coordinates are normalized to [0, 1] relative to image dimensions
                fprintf(fileID, '%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\n', pic_idx,...
                    xywh(1)/w, xywh(2)/h, xywh(3)/w, xywh(4)/h, 0, 0, 0, 0);
            end
        end

        % ---- Add Gaussian thermal noise to the composite image ----
        mean_noise = 0;           % noise mean
        std_dev_noise = 1.5;      % noise standard deviation
        noise = abs(std_dev_noise * randn(size(bg_image)) + mean_noise);
        out_img = bg_image + noise;

        % Save the composite image and close the label file
        imwrite(uint8(out_img), [im_path, num2str(pic_num) '.png']);
        fclose(fileID);
    end
end


%% Verify generated data
% Quick visualization to check that the generated images and labels are correct.
check_idx = 15;
pic_folder = './dataset/';
data_name = ['train_moni_2000_200/'];
im_path = [pic_folder, 'images/', data_name];
lb_path = [pic_folder, 'labels/', data_name];

img_test = imread([im_path, num2str(check_idx) '.png']);
txt_test = load([lb_path, num2str(check_idx) '.txt']);
figure(1), imshow(img_test, []); hold on
min_val = codeValueList_int(txt_test(:,1));

% Display class ID and binary code for each marker
text(txt_test(:,2).*[1280]+3, txt_test(:,3).*[1024], num2str((((txt_test(:,1))))), 'color', 'r', 'FontSize', 22)
text(txt_test(:,2).*[1280]+18, txt_test(:,3).*[1024], num2str(dec2bin((min_val), 15)), 'color', 'g', 'FontSize', 22)
plot(txt_test(:,2).*[1280], txt_test(:,3).*[1024], 'r*')
for i=1:size(txt_test,1)
    rectangle('Position', [txt_test(i,2:5).*[1280,1024,1280,1024]] - [txt_test(i,4:5),0,0].*[1280,1024,1280,1024]/2, ...
              'EdgeColor', 'r', 'LineWidth', 2);
end


%% Display 
check_idx = 1;
pic_folder = './dataset/';
data_name = ['val_moni_200_200/'];
im_path = [pic_folder, 'images/', data_name];
lb_path = [pic_folder, 'labels/', data_name];

img_test = imread([im_path, num2str(check_idx) '.png']);
txt_test = load([lb_path, num2str(check_idx) '.txt']);
txt_test_cor = txt_test(:,2:3).*[1280,1024];
% pre_path = load(['./runs/detect/exp2/', num2str(check_idx) '.txt']);
figure(1), imshow(img_test, []); hold on
min_val = codeValueList_int(txt_test(:,1));

text(txt_test(:,2).*[1280]+3, txt_test(:,3).*[1024], num2str((((txt_test(:,1))))), 'color', 'g', 'FontSize', 22)
text(txt_test(:,2).*[1280]+18, txt_test(:,3).*[1024], num2str(dec2bin((min_val), 15)), 'color', 'g', 'FontSize', 22)
plot(txt_test(:,2).*[1280], txt_test(:,3).*[1024], 'r*')
for i=1:size(txt_test,1)
    rectangle('Position', [txt_test(i,2:5).*[1280,1024,1280,1024]] - [txt_test(i,4:5),0,0].*[1280,1024,1280,1024]/2, ...
              'EdgeColor', 'r', 'LineWidth', 2);
end
