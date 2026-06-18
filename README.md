# Cycle-Aware

%% ============================================================  
%  Project: Synthetic Circular Coded Marker Dataset Generator  
%  Author: Xiaojie Zhang  
%  License:  
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
