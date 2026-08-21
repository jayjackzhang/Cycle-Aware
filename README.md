# Cycle-Aware (Initial Version)

This repository contains the initial implementation of Cycle-Aware for circular marker detection, base on YOLOv5.

## Contents

- `generate_matker_easy.m:` synthetic dataset generation code;
- `train.py:` training script for new model weights;
- `utils/loss.py:` implementation of the proposed Cycle-Aware minimal loss strategy;
- `detect.py:` inference script for test data;
- `verfication.m:` evaluation script for computing precision and recall;
  The mixed training dataset and three test scene datasets;
  A pretrained model weight file in `runs/train/train_seg/weights/`.


