%% 生成编制点列表
clear;
codeSize = 15; % 位数
size_ = 128-1; % 正常大小的标志点图像尺寸,奇数
size_half = (size_ + 1) / 2 - 1;

maginify = 11; % 高分辨倍率,奇数
size_high_resolution = size_ * maginify;

% 生成编制点列表
numCodes = 2^codeSize - 2;
oriCodeList = arrayfun(@(x) dec2bin(x, codeSize), 1:numCodes, 'UniformOutput', false);

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

% 去掉重复的值
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

% bin_arrzy=zeros(32766,15);
% for i=1:32766
%     bin_arrzy(i,:)=oriCodeList{i}-'0';
% end
% 
% bar(sum(bin_arrzy,1)/32766)
% xlabel('编码位', 'FontSize', 24);
% ylabel('出现"1"的概率', 'FontSize', 24);
% ylim([0 1]);

return


%% 读取真值
pt_path='train_seg';%所用训练权重
data_set='article';%验证的数据集
data_name='val';%验证数据集名称
show_=0;
offset=0;
cond_tho=0.06;%小标志的直径阈值train_real
minValueList(32767)=32767;
cam_w=1280;
cam_h=1024;


sample_rate=15;
gt_path=['./dataset/' data_set '/labels/' data_name];
gt_dirInfo = dir([gt_path, '/*.txt']);
target_all=[];
target_all{length(gt_dirInfo)}=[];
t_num=0;
for i=1:length(gt_dirInfo)
    target_all{i}=load([gt_dirInfo(i).folder, '/', num2str(i+offset), '.txt']);
    t_num=t_num+size(target_all{i},1);
end

pr_path=['./runs/detect/' pt_path, '/', data_set, '/', data_name];
accuracy=[];
for i0=1:20
    predict_all=[];
    predict_all{length(gt_dirInfo)}=[];
    % predit_dirInfo = dir([pr_path, '/epoch' num2str((i0-1)*50) '/*.txt']);
    for i=1:length(target_all)
         % filename=[pr_path,'/epoch' num2str((i0-1)*10), '/', num2str(i+offset), '.txt'];
        filename=[pr_path,'/best/', num2str(i+offset), '.txt'];
        if exist(filename, 'file') == 2
            predict_all{i}=load(filename);
        end
    end

    %预测
    statistics=zeros(length(predict_all),6);

    for i=1:length(predict_all)
        target=target_all{i};
        target(:,1)=minValueList(target(:,1));%
        % target(:,1)=codeValueList(target(:,1));%2190对应真实编码
        % target(:,1)=codeValueList_map_int(minValueList(target(:,1)));%最小循环编码

        target(:,2:3)=target(:,2:3).*[cam_w,cam_h];
        % target(target(:,4)>100,:)=[];
        statistics(i,1)=size(target,1);
        % statistics(i,5)=sum(target(:,4)>cond_tho);
        if isempty(predict_all{i})
            continue;
        end

        predit=predict_all{i};
        % predit(:,1)=predit(:,1)+1;
        % predit(:,1)=codeValueList(predit(:,1)+1);
        % dele_idx=predit(:,7)<0.9;
        % predit(dele_idx,:)=[]

        value_d=predit(:,1)<1|predit(:,1)>32766;
        predit(~value_d,1)=minValueList(predit(~value_d,1));
        predit(:,2:3)=predit(:,2:3).*[cam_w,cam_h];
        % predit(:,2:3)=(predit(:,2:3).*[cam_w,cam_h]-[0,8.5])/(1280/2448);

        % if show_
        %     img_Show=imread([gt_path, '/../../images/', data_name, '/', num2str(i), '.png']);
        %     figure(1),imshow(img_Show);hold on;
        %     text(predit(:,2),predit(:,3),num2str(predit(:,1)),'Color', 'green','FontSize',15);
        %     for j=1:size(predit,1)
        %         rectangle('Position', [predit(j,2:5)-[predit(j,4:5)/2,0,0]], 'Curvature', [1, 1], 'EdgeColor', 'r', 'LineWidth', 2);
        %     end
        % end


        % 聚类然后筛选出置信度最高的
        epsilon = 5; % 邻域半径
        MinPts = 1; % 每个簇的最小点数
        % 使用DBSCAN进行聚类
        [idx, www] = dbscan(predit(:,2:3), epsilon, MinPts);
        predit_fine=zeros(max(idx),size(predit,2));
        for ii=1:max(idx)
            idx_ii=find(idx==ii);
            % if (length(idx_ii)<3)
            %     continue;
            % end
            predit_ii=predit(idx_ii,:);
            
            % % 方法零 取置信度最高的
            % [~,idx_2]=max(sum(predit_ii(:,6:7),2));
            % predit_ii_most=predit_ii(predit_ii(:,1)==predit_ii(idx_2,1),:);
            % predit_fine(ii,:)=predit_ii_most(1,:);
            % predit_fine(ii,2:5)=mean(predit_ii_most(:,2:5));
            
            
            % 方法一 取重复次数最多的
            most_frequent_value = mode(predit_ii(:,1));
            predit_ii_most=predit_ii(predit_ii(:,1)==most_frequent_value,:);
            % if mean(predit_ii_most(:,6)<0.8)
            %        continue;
            % end
            predit_fine(ii,:)=predit_ii_most(1,:);
            predit_fine(ii,2:5)=mean(predit_ii_most(:,2:5));

            % % 方法二 提圆矫正采样相关
            % bbox=mean(predit_ii(:,2:5),1);
            % [sampled_values,t_img,x_coords,y_coords] = rect_sample(img,bbox,codeSize,sample_rate);
            % % imshow(img),hold on,plot(bbox(1),bbox(2),'r*')
            % % imshow(t_img,[]),hold on,plot(x_coords',y_coords','r.')
            % if isempty(sampled_values)
            %     % disp([num2str(i0),'次',num2str(i),'张',num2str(ii),'个'])
            %     continue
            % end
            % % sampled_values=sampled_values-(max(sampled_values)+min(sampled_values))/2;
            % [counts, edges] = histcounts(sampled_values, codeSize); 
            % sampled_values=sampled_values-edges(7);
            % unique_vals=unique(predit_ii(:,1));
            % if length(unique_vals)==1
            %     predit_fine(ii,1)=unique_vals;
            % else
            %     correlation_value=0;
            %     for iii=1:length(unique_vals)
            %         binary_string = dec2bin(unique_vals(iii), codeSize);
            %         binary_values = double(binary_string) - '0'; 
            %         repeated_array = repelem(binary_values, sample_rate)-0.5;
            %         [cross_corr, lags] = xcorr(repeated_array, sampled_values);
            %         if max(cross_corr)>correlation_value
            %             correlation_value=max(cross_corr);
            %             predit_fine(ii,1)=unique_vals(iii);
            %         end
            %     end
            % end
            % predit_fine(ii,2:5)=bbox;

        end

        predit_fine(predit_fine(:,1)==0,:)=[];
        [k_idx,dis]=knnsearch(predit_fine(:,2:3),target(:,2:3));
        valid_idx=dis<epsilon;
        pre_idx=k_idx(valid_idx);
        matched=(predit_fine(pre_idx,1)==target(valid_idx,1));        
        predit_match=predit_fine(pre_idx(matched),:);

        statistics(i,1);                                               %真正有这么多个标志点
        statistics(i,6)=size(predit_fine,1);%：                         识别出了这么多个标志点
        statistics(i,2)=sum(valid_idx);%位置匹配的点
        statistics(i,3)=sum(matched);%位置和编号都匹配的点
        statistics(i,4)=size(predit_fine,1)-length(pre_idx);%额外找到的点
        % statistics(i,6)=sum(target(matched,4)>cond_tho);

        
        % if (statistics(i,4))>0
        %     img_Show=imread([gt_path, '/../../images/', data_name, '/', num2str(i+offset), '.png']);
        %     figure(3),imshow(img_Show);hold on;
        % 
        %         for ij=1:size(predit_fine,1)
        %             if ~ismember(ij, pre_idx)
        %                 rectangle('Position', [predit_fine(ij,2:3)-25,50,50], 'Curvature', [0.5, 0.5], 'EdgeColor', 'b', 'LineWidth', 2);
        %                 dec2bin(predit_fine(ij,1))
        %             end
        %         end
        % end


 % show_=1;
        if show_
            % img=imread([gt_path, '/../../images/', data_name, '/', num2str(i), '.png']);
            img_Show=imread([gt_path, '/../../images/', data_name, '/', num2str(i+offset), '.png']);
            figure(3),imshow(img_Show);hold on;
            text(predit_match(:,2),predit_match(:,3),num2str(predit_match(:,1)),'Color', 'green','FontSize',15);

            if  statistics(i,2)<statistics(i,1) %存在标志点区域没检测到~all(valid_idx)
                err_idx=find(~valid_idx);
                for ij=1:length(err_idx)
                    rectangle('Position', [target(err_idx(ij),2:3)-10,20,20], 'Curvature', [1, 1], 'EdgeColor', 'r', 'LineWidth', 2);
                end
            end

            if statistics(i,3)<statistics(i,2)  %存在检测到解码与真值不同~all(matched)
                diff_idx_p=pre_idx(~matched);
                diff_t=find(valid_idx);
                diff_idx_t=diff_t(~matched);
                for ii=1:length(diff_idx_p)
                    plot(predit_fine(diff_idx_p(ii),2),predit_fine(diff_idx_p(ii),3),'*r')
                    text(predit_fine(diff_idx_p(ii),2),predit_fine(diff_idx_p(ii),3),...
                        num2str(predit_fine(diff_idx_p(ii),1)),'Color', 'red','FontSize',15);
                    text(target(diff_idx_t(ii),2),target(diff_idx_t(ii),3)-20,...
                        num2str(target(diff_idx_t(ii),1)),'Color', 'yellow','FontSize',15);
                end
            end

            if statistics(i,4)>0 %存在额外找到的点
                for ij=1:size(predit_fine,1)
                    if ~ismember(ij, pre_idx)
                        rectangle('Position', [predit_fine(ij,2:3)-25,50,50], 'Curvature', [0.5, 0.5], 'EdgeColor', 'b', 'LineWidth', 2);
                    end
                end
                a=1;
            end
            hold off;
        end
    end
    accuracy(i0,1)=sum(statistics(:,3))/sum(statistics(:,1));
    accuracy(i0,2)=sum(statistics(:,2))/sum(statistics(:,1));
    % accuracy(i0,3)=sum(statistics(:,6))/sum(statistics(:,5));

    accuracy(i0,4)=sum(statistics(:,4));
    % accuracy(i0,5)=sum(statistics(:,2))/sum(statistics(:,5));

    Precision = sum(statistics(:,3)) / (sum(statistics(:,6)))  % 
    Recall_det = sum(statistics(:,3)) / sum(statistics(:,1))% TP/N_gt
end

 % writematrix(accuracy(:,1:2), [pt_path,'_II_', data_name,'.txt'], 'delimiter', '\t');

return

%% 真图
imshow(imread('E:\ZXJ\yolo5_marker\datasets\zoulang\images\train_moni\5503.png'))
txt_=load('E:\ZXJ\yolo5_marker\datasets\zoulang\labels\train_moni\5503.txt');hold on
txt_(:,2:3)=txt_(:,2:3).*[1280,1024];
plot(txt_(:,2)+1,txt_(:,3)+1,'r*')
text(txt_(:,2)+5,txt_(:,3)+1,num2str(txt_(:,1)),"Color",'r')
plot(txt_(11,2)+1,txt_(11,3)+1,'g*')
text(txt_(11,2)+5,txt_(11,3)+1,num2str(txt_(11,1)),"Color",'g')
% dlmwrite('E:\ZXJ\yolo5_marker\datasets\real\labels\val\2.txt', txt_, 'delimiter', '\t', 'precision', 10);
%% 测试
% 圆的参数
r = 1; % 圆的半径
center = [0, 0]; % 圆心坐标
a = 2; % 长轴
b = 1; % 短轴
theta = pi/6; % 倾斜角度，单位是弧度（例如30度）

% 计算等分点的角度
num_points = codeSize*sample_rate;
angles = linspace(0, 2*pi, num_points);
% 计算椭圆上的点
ellipse_points = zeros(num_points, 2); % 用于存储椭圆上的点
ellipse_points(:, 1) = center(1) + a * cos(angles) * cos(theta) - b * sin(angles) * sin(theta);
ellipse_points(:, 2) = center(2) + a * cos(angles) * sin(theta) + b * sin(angles) * cos(theta);
sampled_values = interp2(img, ellipse_points(:, 1), ellipse_points(:, 2), 'linear');
[~, edges] = histcounts(sampled_values, codeSize); 
[~,man_idx]=max(diff(sampled_values));
sampled_binary=sampled_values>edges(floor(codeSize/2));
changeIndex = NaN;
% 遍历数组，找出第一个跳变的位置
for i = 2:length(sampled_binary)
    if sampled_binary(i) ~= sampled_binary(i-1)
        changeIndex = i;
        break;
    end
end
if isnan(changeIndex)
    disp('没有跳变');
    return
end
code_idx=changeIndex+(0:14)*15+7;
ex_range=code_idx>num_points;
code_idx(ex_range)=code_idx(ex_range)-num_points;
predit_binary=sampled_binary(code_idx)


% 可视化结果
figure;
imshow(img,[])
hold on;
plot(ellipse_points(:, 1), ellipse_points(:, 2), 'r.', 'LineWidth', 1.5);
axis equal;
title('圆变换为椭圆');
xlabel('X');
ylabel('Y');
hold off;