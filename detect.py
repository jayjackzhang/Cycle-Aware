# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""
Run YOLOv5 detection inference on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources:
    $ python detect.py --weights yolov5s.pt --source 0                               # webcam
                                                     img.jpg                         # image
                                                     vid.mp4                         # video
                                                     screen                          # screenshot
                                                     path/                           # directory
                                                     list.txt                        # list of images
                                                     list.streams                    # list of streams
                                                     'path/*.jpg'                    # glob
                                                     'https://youtu.be/LNwODJXcvt4'  # YouTube
                                                     'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python detect.py --weights yolov5s.pt                 # PyTorch
                                 yolov5s.torchscript        # TorchScript
                                 yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                 yolov5s_openvino_model     # OpenVINO
                                 yolov5s.engine             # TensorRT
                                 yolov5s.mlmodel            # CoreML (macOS-only)
                                 yolov5s_saved_model        # TensorFlow SavedModel
                                 yolov5s.pb                 # TensorFlow GraphDef
                                 yolov5s.tflite             # TensorFlow Lite
                                 yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                                 yolov5s_paddle_model       # PaddlePaddle
"""

import argparse
import csv
import os
import platform
import sys
from pathlib import Path, WindowsPath

import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from ultralytics.utils.plotting import Annotator, colors, save_one_box

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (
    LOGGER,
    Profile,
    check_file,
    check_img_size,
    check_imshow,
    check_requirements,
    colorstr,
    cv2,
    increment_path,
    non_max_suppression,
    print_args,
    scale_boxes,
    strip_optimizer,
    xyxy2xywh,
)
from utils.torch_utils import select_device, smart_inference_mode
import numpy as np
import scipy.io


class FeatureExtractor:
    def __init__(self, model):
        self.model = model
        self.features = {}
        self.hooks = []

        # 注册所有卷积层的钩子
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                self.hooks.append(module.register_forward_hook(
                    lambda m, inp, out, name=name: self.features.update({name: out})
                ))

    def __call__(self, x):
        self.features.clear()
        with torch.no_grad():
            outputs = self.model(x)
        return outputs, self.features

@smart_inference_mode()
def run(
    weights=ROOT / "yolov5s.pt",  # model path or triton URL
    source=ROOT / "data/images",  # file/dir/URL/glob/screen/0(webcam)
    data=ROOT / "data/coco128.yaml",  # dataset.yaml path
    imgsz=(640, 640),  # inference size (height, width)
    conf_thres=0.5,  # confidence threshold
    iou_thres=0.45,  # NMS IOU threshold
    max_det=1000,  # maximum detections per image
    device="",  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    view_img=False,  # show results
    save_txt=True,  # save results to *.txt
    save_csv=False,  # save results in CSV format
    save_conf=False,  # save confidences in --save-txt labels
    save_crop=False,  # save cropped prediction boxes
    nosave=False,  # do not save images/videos
    classes=None,  # filter by class: --class 0, or --class 0 2 3
    agnostic_nms=False,  # class-agnostic NMS
    augment=False,  # augmented inference
    visualize=False,  # visualize features
    update=False,  # update all models
    project=ROOT / "runs/detect",  # save results to project/name
    name="exp",  # save results to project/name
    exist_ok=False,  # existing project/name ok, do not increment
    line_thickness=3,  # bounding box thickness (pixels)
    hide_labels=False,  # hide labels
    hide_conf=False,  # hide confidences
    half=False,  # use FP16 half-precision inference
    dnn=False,  # use OpenCV DNN for ONNX inference
    vid_stride=1,  # video frame-rate stride
):
    source = str(source)
    save_img = not nosave and not source.endswith(".txt")  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(("rtsp://", "rtmp://", "http://", "https://"))
    webcam = source.isnumeric() or source.endswith(".streams") or (is_url and not is_file)
    screenshot = source.lower().startswith("screen")
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    source_p = WindowsPath(source)
    if len(weights.parts) == 5:
        save_dir = Path(project) / weights.parts[2] / source_p.parts[2] / source_p.parts[3] / weights.stem
        save_dir.mkdir(parents=True, exist_ok=True)
    else:
        save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
        (save_dir / "labels" if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    checkpoint = torch.load(str(weights), map_location=device)
    # 直接提取 CosFace 参数
    if 'cosface_weight' in checkpoint and 'bn_embed' in checkpoint:
        # 1. 提取类中心矩阵
        cosface_weight = checkpoint['cosface_weight'].to(device)  # [100, 15]
        # 2. 重建 BN 层并加载状态
        embed_dim = cosface_weight.shape[1]  # 15
        bn_embed = torch.nn.BatchNorm1d(embed_dim, affine=False).to(device)
        bn_embed.load_state_dict(checkpoint['bn_embed'])
        bn_embed.eval()  # 固定 BN 统计量
        print("bn running_mean:", bn_embed.running_mean[:5])
        print("bn running_var:", bn_embed.running_var[:5])
        print("cosface_weight shape:", cosface_weight.shape)
        print("cosface_weight[:3, :5]:", cosface_weight[:3, :5])
        w = torch.nn.functional.normalize(cosface_weight, dim=1)
        sim = w @ w.T  # [100,100]
        print(sim[49].topk(5))  # 49 和哪些类最相似
        print(f"Loaded CosFace params: weight={cosface_weight.shape}, bn_dim={embed_dim}")
    else:
        cosface_weight = None
        bn_embed = None
        print("未找到 CosFace 参数，将跳过")

    # # 打印所有层及其参数量
    # for name, module in model.named_modules(): # 全连接层在 PyTorch 中是 nn.Linear
    #     weight_params = module.weight.numel()
    #     bias_params = module.bias.numel() if module.bias is not None else 0
    #     total_params = weight_params + bias_params
    #     print(f"Layer: {name} | Parameters: {total_params} (W: {weight_params}, B: {bias_params})")

    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    bs = 16  # batch_size
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(device=device), Profile(device=device), Profile(device=device))
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim
            if model.xml and im.shape[0] > 1:
                ims = torch.chunk(im, im.shape[0], 0)

        # Inference
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            if model.xml and im.shape[0] > 1:
                pred = None
                for image in ims:
                    if pred is None:
                        pred = model(image, augment=augment, visualize=visualize).unsqueeze(0)
                    else:
                        pred = torch.cat((pred, model(image, augment=augment, visualize=visualize).unsqueeze(0)), dim=0)
                pred = [pred, None]
            else:
                # # 创建特征提取器
                # extractor = FeatureExtractor(model)
                # outputs, features = extractor(im,)
                #
                # pp1=0
                # for name, feat in features.items():
                #     save_dir = os.path.join('pic')
                #     pp1=pp1+1
                #     pp2=0
                #     for pici in range(min(20, feat.shape[1])):  # 只保存前10个通道
                #         # 提取单个通道的特征图
                #         pp2 = pp2 + 1
                #         channel_data = feat[0, pici, :, :].cpu().numpy()
                #         # 归一化到 [0, 1]（可选）
                #         channel_data_normalized = (channel_data - channel_data.min()) / (
                #                     channel_data.max() - channel_data.min())
                #         # 保存为 .mat 文件
                #         save_path = os.path.join(save_dir, f'{pp1}_channel_{pp2}.mat')
                #         scipy.io.savemat(save_path, {'tensor': channel_data_normalized})
                # pred = model(im.expand(10, -1, -1, -1), augment=augment, visualize=visualize)
                pred, train_out, = model(im, augment=augment, visualize=visualize)
        # NMS
        with dt[2]:

            pred = non_max_suppression(pred, train_out, 0.7, 0.7, classes, agnostic_nms, max_det=3000,cosface_weight=cosface_weight, bn_embed=bn_embed)

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Define the path for the CSV file
        csv_path = save_dir / "predictions.csv"

        # Create or append to the CSV file
        def write_to_csv(image_name, prediction, confidence):
            """Writes prediction data for an image to a CSV file, appending if the file exists."""
            data = {"Image Name": image_name, "Prediction": prediction, "Confidence": confidence}
            with open(csv_path, mode="a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=data.keys())
                if not csv_path.is_file():
                    writer.writeheader()
                writer.writerow(data)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f"{i}: "
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, "frame", 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg

            if len(weights.parts) == 5:
                txt_path = str(save_dir / p.stem) + (
                    "" if dataset.mode == "image" else f"_{frame}")  # im.txt
            else:
                txt_path = str(save_dir / "labels" / p.stem) + ("" if dataset.mode == "image" else f"_{frame}")  # im.txt


            s += "%gx%g " % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                # det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                # for c in det[:, 5].unique():
                #     n = (det[:, 5] == c).sum()  # detections per class
                #     s += f"{n} {[int(c)]}{'s' * (n > 1)}, "  # add to string
                det = det.to('cpu')
                with open(f"{txt_path}.txt", "w") as f:
                    # Write results
                    for det_data in reversed(det):
                        xyxy = det_data[:4]
                        conf = det_data[4:5]
                        cls = det_data[5:6]
                        conf2 = det_data[6]
                        c = int(cls)  # integer class
                        label = [c] if hide_conf else f"{[c]}"
                        confidence = float(conf)
                        confidence_str = f"{confidence:.2f}"

                        if save_csv:
                            write_to_csv(p.name, label, confidence_str)

                        if save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            line = (cls, *xywh, conf, conf2) if save_conf else (cls, *xywh)  # label format
                            if len(line)==7:
                                f.write(("%g " * len(line)).rstrip() % line + "\n")

                        if save_img or save_crop or view_img:  # Add bbox to image
                            c = int(cls)  # integer class
                            label = None if hide_labels else ([c] if hide_conf else f"{[c]} {conf.item():.2f}")
                            annotator.box_label(xyxy, label, color=colors(c, True))
                        if save_crop:
                            save_one_box(xyxy, imc, file=save_dir / "crops" / [c] / f"{p.stem}.jpg", BGR=True)

            # Stream results
            im0 = annotator.result()
            if view_img:
                if platform.system() == "Linux" and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                # cv2.imshow(str(p), im0)
                # cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == "image":
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix(".mp4"))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                    vid_writer[i].write(im0)

        # Print time (inference-only)
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

    # Print results
    t = tuple(x.t / seen * 1e3 for x in dt)  # speeds per image
    LOGGER.info(f"Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}" % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ""
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)


def parse_opt():
    """Parses command-line arguments for YOLOv5 detection, setting inference options and model configurations."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", nargs="+", type=str, default=ROOT / "runs/train/train_seg/weights/best.pt", help="model path or triton URL")
    parser.add_argument("--source", type=str, default=ROOT / "./dataset/article/images/val", help="file/dir/URL/glob/screen/0(webcam)")
    # parser.add_argument("--source", type=str, default=ROOT / "D:/ZXJ_play//marker_rec/Marker_M/datasets/real_cap", help="file/dir/URL/glob/screen/0(webcam)")

    # parser.add_argument("--source", type=str, default="C:/Users/zxj/Desktop/ddd", help="file/dir/URL/glob/screen/0(webcam)")
    parser.add_argument("--data", type=str, default=ROOT / "data/Code_Marker.yaml", help="(optional) dataset.yaml path")
    parser.add_argument("--imgsz", "--img", "--img-size", nargs="+", type=int, default=[1280], help="inference size h,w")
    parser.add_argument("--conf-thres", type=float, default=0.5, help="confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--max-det", type=int, default=1000, help="maximum detections per image")
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--view-img", action="store_true", default=True, help="show results")
    parser.add_argument("--save-txt", action="store_true", default=True, help="save results to *.txt")
    parser.add_argument("--save-csv", action="store_true", help="save results in CSV format")
    parser.add_argument("--save-conf", action="store_true", default=True,  help="save confidences in --save-txt labels")
    parser.add_argument("--save-crop", action="store_true", default=False, help="save cropped prediction boxes")
    parser.add_argument("--nosave", action="store_true", default=True, help="do not save images/videos")
    parser.add_argument("--classes", nargs="+", type=int, help="filter by class: --classes 0, or --classes 0 2 3")
    parser.add_argument("--agnostic-nms", action="store_true", help="class-agnostic NMS")
    parser.add_argument("--augment", action="store_true", help="augmented inference")
    parser.add_argument("--visualize", action="store_true", help="visualize features")
    parser.add_argument("--update", action="store_true", help="update all models")
    parser.add_argument("--project", default=ROOT / "runs/detect", help="save results to project/name")
    parser.add_argument("--name", default="exp", help="save results to project/name")
    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")
    parser.add_argument("--line-thickness", default=3, type=int, help="bounding box thickness (pixels)")
    parser.add_argument("--hide-labels", default=False, action="store_true", help="hide labels")
    parser.add_argument("--hide-conf", default=False, action="store_true", help="hide confidences")
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    parser.add_argument("--dnn", action="store_true", help="use OpenCV DNN for ONNX inference")
    parser.add_argument("--vid-stride", type=int, default=1, help="video frame-rate stride")
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    """Executes YOLOv5 model inference with given options, checking requirements before running the model."""
    check_requirements(ROOT / "requirements.txt", exclude=("tensorboard", "thop"))
    run(**vars(opt))
    # for i in range(20):
    #      if i == 19:
            # opt.weights = ROOT / "runs/train/train_seg_ellip_1/weights/epoch" f"{i*10}.pt"
            # opt.weights = ROOT / "runs/train/train_seg_ellip_2/weights/epoch" f"{i*10}.pt"
            # opt.source = ROOT / "../datasets/zoulang/images/val_real"
            # opt.source = ROOT / "../datasets/test/images/test"
            # # opt.weights = ROOT / "runs/train/train_seg/weights/best.pt"
            # run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
