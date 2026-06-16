%% ============================================================
%  Project: Synthetic Circular Coded Marker Dataset Generator
%  Author: [Your Name]
%  License: [Add your license, e.g. MIT / GPL / Academic Use]
%
%  ------------------------------------------------------------
%  ⚠ IMPORTANT DATA DEPENDENCY NOTICE
%  ------------------------------------------------------------
%  This script uses the COCO dataset as background images:
%      Microsoft COCO dataset (2017)
%      https://cocodataset.org/
%
%  If COCO dataset is NOT found locally:
%      → The script will automatically fall back to BLACK background
%      → No crash will occur
%      → Generated dataset will NOT contain natural image embedding
%
%  To reproduce full results:
%      1. Download COCO 2017 train/val images
%      2. Place under: ./datasets/coco/images/
%
%  ------------------------------------------------------------
%  Function:
%  - Generate cyclic binary coded circular markers
%  - Simulate perspective distortion + photometric variation
%  - Embed markers into COCO or synthetic black background
%  - Output YOLO-style labels
% ============================================================

clear; clc;

%% ===================== 1. CODE GENERATION =====================

codeSize = 15;                 % binary code length
size_m = 55 * 5;                % template resolution
size_half = (size_m + 1)/2 - 1;

numCodes = 2^codeSize - 2;
oriCodeList = arrayfun(@(x) dec2bin(x, codeSize), ...
    1:numCodes, 'UniformOutput', false);

minValueList = zeros(1, numCodes);

for i = 1:numCodes
    tempCode = oriCodeList{i};
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

%% remove duplicate cyclic-equivalent codes
codeList = {};
codeValueList = [];

numDiff = 0;

for i = 1:numCodes
    isExist = any(minValueList(i) == codeValueList);

    if ~isExist
        numDiff = numDiff + 1;
        codeValueList(numDiff) = minValueList(i);
        codeList{numDiff} = oriCodeList{i};
    end
end

%% ===================== 2. MARKER RENDERING =====================

size_high_resolution = size_m;

[X, Y] = meshgrid(1:size_high_resolution, 1:size_high_resolution);

cen_x = (size_high_resolution+1)/2;
cen_y = cen_x;

distance_from_center = sqrt((X - cen_x).^2 + (Y - cen_y).^2);
ang_from_center = atan2((X - cen_x), (Y - cen_y));

point_size_time = 2 * (cen_x-1) / 64;

r0   = point_size_time * 10 / 2;
lb   = r0/2.5 * 27;
rin  = r0/2.5 * 6.5;
rout = r0/2.5 * 10;

%% angular sector encoding
ang_mask = zeros(size_high_resolution);
angle_step = 2*pi/codeSize;
sector_angle_step = -pi:angle_step:pi;

for i = 1:codeSize
    ang_mask(ang_from_center > sector_angle_step(i)) = i;
end

%% output folder
marker_path = './datasets/oripic/';
tra_path = [marker_path '/'];

if exist(marker_path, 'dir') ~= 7
    mkdir(marker_path);
    mkdir(tra_path);
end

%% draw markers
for i = 1:length(codeList)

    img = zeros(size_high_resolution);

    img(distance_from_center <= rout) = 1;

    zros_idx = find(codeList{i} == '0');
    for j = 1:length(zros_idx)
        img(ang_mask == zros_idx(j)) = 0;
    end

    img(distance_from_center <= rin) = 0;
    img(distance_from_center <= r0) = 1;

    imwrite(uint8(img*255), [tra_path num2str(i) '.png']);
end

%% ===================== 3. DATASET SETTINGS =====================

mode_generate = 'val';
pic_total = 10;
template_total = 2190;

data_name = [mode_generate '_moni_' num2str(pic_total) '_' num2str(template_total) '/'];

%% ===================== 4. COCO BACKGROUND LOADING =====================
% If COCO is missing → fallback to BLACK background

coco_path = ['./datasets/coco/images/' mode_generate '2017'];

useCOCO = exist(coco_path, 'dir') == 7;

if useCOCO
    imds = imageDatastore(coco_path);

    augmenter = imageDataAugmenter( ...
        'RandRotation', [-10, 10], ...
        'RandXTranslation', [-20, 20], ...
        'RandYTranslation', [-20, 20], ...
        'RandXScale', [0.8, 1.2], ...
        'RandYScale', [0.8, 1.2], ...
        'RandXReflection', true, ...
        'RandYReflection', true);

    augimds = augmentedImageDatastore([1024, 1280], imds, ...
        'DataAugmentation', augmenter, ...
        'ColorPreprocessing', 'rgb2gray');

    augimds.MiniBatchSize = 1;

    coco_data = cell(pic_total,1);

    reset(augimds);
    for i = 1:pic_total
        coco_data{i} = read(augimds);
    end
else
    warning('COCO dataset NOT found. Using BLACK background instead.');
    coco_data = cell(pic_total,1);
    for i = 1:pic_total
        coco_data{i}.input{1,1} = zeros(1024, 1280);
    end
end

%% ===================== 5. SYNTHESIS PARAMETERS =====================

w = 1280;
h = 1024;

p_min = 30;
p_max = 180;

d = 5500;

ax_range = [-75, 75];
az_range = [0, 360];

marker_folder = './datasets/Synthesis/';
im_path = [marker_folder 'images/' data_name];
lb_path = [marker_folder 'labels/' data_name];

if exist(im_path, 'dir') ~= 7
    mkdir(im_path);
    mkdir(lb_path);
end

patch_half = 0.5;

%% ===================== 6. GENERATION =====================

for pic_num=1:pic_total
    bg_image = double(coco_data{pic_num}.input{1,1});
    % if isfile([im_path, num2str(pic_num) '.png'])
    %     continue;
    % end
    if mode_generate=="train"
        rng(pic_num * 1000 + feature('getpid'));
    else
        rng((pic_num+10000) * 1000 + feature('getpid'));
    end    
    para_all = rand(100,9);
    mask_all=zeros(h,w);

    fileID = fopen([lb_path, num2str(pic_num) '.txt'], 'w');
    for i=1:100
        % 模板读取与预处理
        pic_idx=floor(template_total * para_all(i,1)) + 1;
        template = imread(['./datasets/oripic/' num2str(pic_idx) '.png']);
        template = double(template);                % 转为双精度便于运算
        template(template==0) = para_all(i,2)*10;   % 标志点黑色部分存在一定反射率
        template = template * (para_all(i,3)*0.5 + 0.5); % 白色部分曝光调整
        template = uint8(template);                 % 转回 uint8

        % 模板边长
        m = size(template, 1);
        assert(size(template,2) == m, 'Template must be square');

        % 随机生成旋转角度（绕Z和绕X）
        gamma = az_range(1) + para_all(i,4)*(az_range(2)-az_range(1));
        alpha = ax_range(1) + para_all(i,5)*(ax_range(2)-ax_range(1));

        % 旋转矩阵
        Rz = [cosd(gamma), -sind(gamma), 0;
            sind(gamma),  cosd(gamma), 0;
            0,            0,           1];
        Rx = [1, 0,            0;
            0, cosd(alpha), -sind(alpha);
            0, sind(alpha),  cosd(alpha)];
        R = Rx * Rz;

        % 距离与平移参数
        t_z_max = d * (m/p_min - 1);
        t_z_min = d * (m/p_max - 1);
        t_z = t_z_min + para_all(i,6)*(t_z_max - t_z_min);
        scale_t=d / (d + t_z);
        l_proj = m * scale_t;   % 投影边长（像素）

        % 随机平移，保证标志完全落在画布内
        tx_min = (l_proj/2 - w/2) * (d + t_z)/d;
        tx_max = (w - l_proj/2 - w/2) * (d + t_z)/d;
        ty_min = (l_proj/2 - h/2) * (d + t_z)/d;
        ty_max = (h - l_proj/2 - h/2) * (d + t_z)/d;
        t_x = tx_min + para_all(i,7)*(tx_max - tx_min);
        t_y = ty_min + para_all(i,8)*(ty_max - ty_min);
        % 构造单应矩阵 H
        H = [d*R(1,1)+(w+1)/2*R(3,1), d*R(1,2)+(w+1)/2*R(3,2), d*t_x+(w+1)/2*(d+t_z);
            d*R(2,1)+(h+1)/2*R(3,1), d*R(2,2)+(h+1)/2*R(3,2), d*t_y+(h+1)/2*(d+t_z);
            R(3,1),              R(3,2),              d+t_z] * ...
            [1, 0, -(m+1)/2;
            0, 1, -(m+1)/2;
            0, 0,  1];
        H = H / H(3,3);   % 归一化
        % -----------------------------
        % 1. 新图四角 + 中心
        % -----------------------------
        [h_pad, w_pad] = size(template);
        corners = [1, 1, 1;  1, h_pad, 1; w_pad, h_pad, 1;w_pad, 1, 1;(m+1)/2,(m+1)/2,1];
        corners_t = H * corners' ;
        corners_t = corners_t ./ corners_t(3,:);
        xywh=[corners_t(1,5),  corners_t(2,5),l_proj,l_proj];%中心坐标长宽

        % -----------------------------
        % 2. 新图四边形 mask（需要插值的像素）
        % -----------------------------
        mask = poly2mask(corners_t(1,1:4), corners_t(2,1:4), h, w);
        [ydst, xdst] = find(mask);

        % -----------------------------
        % 3. 计算参考像素对应原图四边形区域
        % -----------------------------
        Hinv = inv(H);
        refX=xywh(1);refY=xywh(2);
        corners = [ ...
            refX-0.5, refX+0.5, refX+0.5, refX-0.5;
            refY-0.5, refY-0.5, refY+0.5, refY+0.5;
            1,       1,       1,       1];
        src = Hinv * corners;
        src(1,:) = src(1,:) ./ src(3,:);
        src(2,:) = src(2,:) ./ src(3,:);
        maskSrc = poly2mask(src(1,:), src(2,:), h, w);
        [ys, xs] = find(maskSrc);
        dx = xs -(m+1)/2;
        dy = ys -(m+1)/2;

        % -----------------------------
        % 3) 对新图每个需要插值的像素，平移同一组mask偏移并求平均
        % -----------------------------
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

        warped = imgaussfilt(warped, 1*para_all(i,9));  % 高斯模糊


        if sum(mask_all(mask))>1%去掉重合标志点
            continue;
        else
            bg_image(mask)=warped(mask);
            mask_all(mask)=1;

            fprintf(fileID, '%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\n', pic_idx,...
                xywh(1)/w,xywh(2)/h,xywh(3)/w,xywh(4)/h,0,0,0,0);
        end

    end
    % % 显示结果
    % figure;
    % imshow(bg_image,[]); hold on;
    % title('合成图像');

    % 高斯分布的热噪声
    mean_noise = 0; % 噪声均值
    std_dev_noise = 1.5; % 噪声标准差
    noise = abs(std_dev_noise * randn(size(bg_image)) + mean_noise);
    out_img=bg_image+noise;

    imwrite(uint8(out_img),[im_path, num2str(pic_num) '.png']);
    fclose(fileID);
end

%% ===================== END =====================