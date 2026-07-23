# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""Loss functions."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.metrics import bbox_iou
from utils.torch_utils import de_parallel
import scipy.io as sio

import sys
def smooth_BCE(eps=0.1):
    """Returns label smoothing BCE targets for reducing overfitting; pos: `1.0 - 0.5*eps`, neg: `0.5*eps`. For details see https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441"""
    return 1.0 - 0.5 * eps, 0.5 * eps

import math

class BCEBlurWithLogitsLoss(nn.Module):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self, alpha=0.05):
        """Initializes a modified BCEWithLogitsLoss with reduced missing label effects, taking optional alpha smoothing
        parameter.
        """
        super().__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction="none")  # must be nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred, true):
        """Computes modified BCE loss for YOLOv5 with reduced missing label effects, taking pred and true tensors,
        returns mean loss.
        """
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)  # prob from logits
        dx = pred - true  # reduce only missing label effects
        # dx = (pred - true).abs()  # reduce missing label and false label effects
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        """Initializes FocalLoss with specified loss function, gamma, and alpha values; modifies loss reduction to
        'none'.
        """
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = "none"  # required to apply FL to each element

    def forward(self, pred, true):
        """Calculates the focal loss between predicted and true labels using a modified BCEWithLogitsLoss."""
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:  # 'none'
            return loss


class QFocalLoss(nn.Module):
    # Wraps Quality focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        """Initializes Quality Focal Loss with given loss function, gamma, alpha; modifies reduction to 'none'."""
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = "none"  # required to apply FL to each element

    def forward(self, pred, true):
        """Computes the focal loss between `pred` and `true` using BCEWithLogitsLoss, adjusting for imbalance with
        `gamma` and `alpha`.
        """
        loss = self.loss_fcn(pred, true)

        pred_prob = torch.sigmoid(pred)  # prob from logits
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:  # 'none'
            return loss


class ComputeLoss(nn.Module):
    sort_obj_iou = False

    # Compute losses
    def __init__(self, model, autobalance=False):
        super().__init__()
        """Initializes ComputeLoss with model and autobalance option, autobalances losses if True."""
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h["cls_pw"]], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h["obj_pw"]], device=device))

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get("label_smoothing", 0.0))  # positive, negative BCE targets

        # Focal loss
        g = h["fl_gamma"]  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        m = de_parallel(model).model[-1]  # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(m.nl, [4.0, 1.0, 0.25, 0.06, 0.02])  # P3-P7
        self.ssi = list(m.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, 1.0, h, autobalance
        self.cross = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h["cls_pw"]], device=device), reduction='none')


        self.l1_loss = nn.L1Loss(reduction='none')
        self.na = m.na  # number of anchors
        self.nc = m.nc  # number of classes
        self.embed_dim = self.nc
        # n个类别的15维原型（可训练）
        self.bn_embed = nn.BatchNorm1d(self.embed_dim, affine=False).to(device)
        self.cosface_weight = nn.Parameter(torch.empty(2190, 15, device=device))
        nn.init.xavier_uniform_(self.cosface_weight)

        data = sio.loadmat('D:/ZXJ_play/marker_rec/codebook.mat')
        self.codeValueList = torch.tensor(data['codeValueList_int'].flatten(), device=device)
        self.codeValueList_map_int = torch.tensor(data['codeValueList_map_int'].flatten(), device=device)
        self.minValueList = torch.tensor(data['minValueList_int'].flatten(), device=device)
        self.criterion = nn.CrossEntropyLoss()
        self.nl = m.nl  # number of layers
        self.anchors = m.anchors
        self.device = device

        with open('./dataset/names_all.txt', 'r') as file:
            # 初始化一个空字典
            data_dict = {}
            # 逐行读取文件内容
            for line in file:
                # 去除每行末尾的换行符并以冒号分割键和值
                key, value = line.strip().split(': ')
                value = value.replace("'", '')
                # 将键值对添加到字典中
                tensor = torch.tensor([int(char) for char in value])
                # 将键值对添加到字典中
                data_dict[int(key)] = tensor
            self.data_dict = data_dict

        with open('./dataset/names_min.txt', 'r') as file:
            # 初始化一个空字典
            data_dict_min = {}
            # 逐行读取文件内容
            for line in file:
                # 去除每行末尾的换行符并以冒号分割键和值
                key, value = line.strip().split(': ')
                # value = value.replace("'", '')
                # # 将键值对添加到字典中
                # tensor = torch.tensor([int(char) for char in value])
                # 将键值对添加到字典中
                data_dict_min[int(key)] = int(value)
            self.data_dict_min = data_dict_min

    def __call__(self, p, targets):  # predictions, targets
        """Performs forward pass, calculating class, box, and object loss for given predictions and targets."""
        lcls = torch.zeros(1, device=self.device)  # class loss
        lbox = torch.zeros(1, device=self.device)  # box loss
        lobj = torch.zeros(1, device=self.device)  # object loss
        # if self.nc<=self.nc:
        targets=targets[:,:6]

        tcls, tbox, indices, anchors, add_imf = self.build_targets(p, targets)  # targets

        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros(pi.shape[:4], dtype=pi.dtype, device=self.device)  # target obj

            n = b.shape[0]  # number of targets
            if n:
                # pxy, pwh, _, pcls = pi[b, a, gj, gi].tensor_split((2, 4, 5), dim=1)  # faster, requires torch 1.8.0
                pxy, pwh, _, pcls = pi[b, a, gj, gi].split((2, 2, 1, self.nc), 1)  # target-subset of predictions

                # Regression
                pxy = pxy.sigmoid() * 2 - 0.5
                pwh = (pwh.sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted bo5x
                iou = bbox_iou(pbox, tbox[i], CIoU=True).squeeze()  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                iou = iou.detach().clamp(0).type(tobj.dtype)
                if self.sort_obj_iou:
                    j = iou.argsort()
                    b, a, gj, gi, iou = b[j], a[j], gj[j], gi[j], iou[j]
                if self.gr < 1:
                    iou = (1.0 - self.gr) + self.gr * iou
                tobj[b, a, gj, gi] = iou  # iou ratio

                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)

                    cal_type = 1
                    # if cal_type == 0:  # 三万类直接对应进行计算
                    #     selected_values = [self.data_dict[ii.item()] for ii in tcls[i]]
                    #     expected = torch.stack(selected_values).type_as(pcls).to('cuda')
                    #     losses = self.cross(pcls[:, :self.nc], expected)
                    #     losses_sum = losses.sum(dim=1)
                    #     lcls += (losses_sum.mean())

                    if cal_type == 1:  #cycle-aware
                        selected_values = [self.data_dict[ii.item()] for ii in tcls[i]]
                        expected = torch.stack(selected_values).type_as(pcls).to('cuda')
                        # 生成一个用于循环位移的索引张量
                        n, m = expected.shape
                        index = torch.arange(m, device=expected.device).unsqueeze(0) + torch.arange(m, device=expected.device).unsqueeze(1)
                        index = index % m
                        # 对每一行进行循环位移并堆叠
                        target_result = expected.unsqueeze(1).expand(-1, m, -1).gather(2,index.unsqueeze(0).expand(n, -1, -1)).reshape(n * m, m)
                        predit_result = (pcls[:, :self.nc].unsqueeze(1).expand(n, m, m).reshape(n * m, m))
                        losses = self.cross(predit_result,target_result)
                        losses_sum = losses.sum(dim=1)
                        # 将损失 reshaped 成 (n, 15, 15) 的形状
                        losses_mean1_15 = losses_sum.view(n, m)
                        # 在每组 15 个元素中选择最小的损失
                        min_losses, min_idx = torch.min(losses_mean1_15, dim=1)
                        # 计算最终损失
                        lcls += (min_losses.mean())

                    if cal_type == 2:  #min-code
                        selected_values = [self.data_dict[self.data_dict_min[ii.item()]] for ii in tcls[i]]
                        expected = torch.stack(selected_values).type_as(pcls).to('cuda')
                        losses = self.cross(pcls[:, :self.nc], expected)
                        losses_sum = losses.sum(dim=1)
                        lcls += (losses_sum.mean())

                    if cal_type == 3:  #ElasticFace
                        # ElasticFace-Cos 超参数
                        cos_scale = 30.0  # 缩放因子 s
                        mean_m = 0.35  # 高斯分布均值（原 CosFace 的 margin）
                        sigma = 0.025  # 高斯分布标准差（弹性程度）
                        # 1. 取分类头前15维作为原始嵌入
                        raw_embed = pcls[:, :self.embed_dim].float()  # [n, 15]
                        # 2. L2 归一化
                        # normed_embed = self.bn_embed(raw_embed)
                        normed_embed = F.normalize(raw_embed, p=2, dim=1)
                        # 3. 归一化类中心矩阵
                        w = F.normalize(self.cosface_weight, p=2, dim=1)  # [100, 15]
                        # 4. 计算余弦相似度
                        cosine = F.linear(normed_embed, w)  # [n, 100]
                        # 5. 为每个样本采样随机 margin
                        idx = self.minValueList[tcls[i] - 1]
                        class_num = self.codeValueList_map_int[idx - 1] - 1
                        class_num = torch.tensor(class_num, device=raw_embed.device).long()
                        random_margin = torch.normal(mean=mean_m, std=sigma, size=(cosine.size(0),),
                                                     device=cosine.device)
                        random_margin = torch.clamp(random_margin, min=0.0)  # margin 非负
                        # 6. one-hot 掩码
                        one_hot = torch.zeros_like(cosine)
                        one_hot.scatter_(1, class_num.view(-1, 1), 1.0)
                        # 7. 在真实类上减去随机 margin
                        cosine_m = cosine - one_hot * random_margin.unsqueeze(1)
                        cosine_m = torch.clamp(cosine_m, -1.0 + 1e-7, 1.0 - 1e-7)  # 数值稳定
                        # 8. 缩放 + 交叉熵损失
                        logits = cosine_m * cos_scale
                        loss_embed = self.criterion(logits, class_num)
                        lcls += loss_embed

                    #
                    # if cal_type == 3:  # 100类15维循环最小
                    #     pred_embed = pcls[:, :self.embed_dim].float()
                    #     class_num = (self.codeValueList[tcls[i]-1]).long()
                    #     n = class_num.shape[0]
                    #     m = 15
                    #     # 转成15位二进制
                    #     bits = ((class_num.unsqueeze(1) >> torch.arange(m - 1, -1, -1,device=class_num.device)) & 1).float()  # (n,15)
                    #     # 生成所有循环位移
                    #     index = (torch.arange(m, device=bits.device).unsqueeze(0)+ torch.arange(m, device=bits.device).unsqueeze(1)) % m
                    #     target_result = (bits.unsqueeze(1).expand(-1, m, -1).gather(2, index.unsqueeze(0).expand(n, -1, -1)).reshape(n * m, m))
                    #     predict_result = (pred_embed.unsqueeze(1).expand(n, m, m).reshape(n * m, m))
                    #     # BCE loss
                    #     losses = self.cross(predict_result,target_result)  # (n*m,15)
                    #     losses_sum = losses.sum(dim=1)
                    #     # 每个样本15个位移中取最小
                    #     losses_reshape = losses_sum.view(n, m)
                    #     min_losses, min_idx = torch.min(losses_reshape,dim=1)
                    #     loss_embed = min_losses.mean()
                    #     lcls += loss_embed
                    #
                    # if cal_type == 4:  #100类100维onthot-CE
                    #     pred_embed = pcls[:, :self.embed_dim].float()
                    #     class_num = (tcls[i] - 1).long()
                    #     loss_embed = self.criterion(pred_embed, class_num)
                    #     lcls += loss_embed
                    #
                    # if cal_type == 5:  #100类15维cosface
                    #     cos_scale = 30.0  # CosFace 尺度因子 s
                    #     cos_margin = 0.35  # 余弦边距 m
                    #     # 1. 取分类头前15维作为原始嵌入
                    #     raw_embed = pcls[:, :self.embed_dim].float()  # [n, 15]
                    #     # 2. BN + L2 归一化
                    #     # normed_embed = self.bn_embed(raw_embed)
                    #     normed_embed = F.normalize(raw_embed, p=2, dim=1)
                    #     # 3. 归一化类中心矩阵
                    #     w = F.normalize(self.cosface_weight, p=2, dim=1)  # [100, 15]
                    #     # 4. 计算余弦相似度
                    #     cosine = F.linear(normed_embed, w)  # [n, 100]
                    #     # 5. CosFace 边距（正确类别的余弦值减去 margin）
                    #     idx = self.minValueList[tcls[i] - 1]
                    #     class_num = self.codeValueList_map_int[idx - 1] - 1
                    #     class_num = torch.tensor(class_num, device=raw_embed.device).long()
                    #     one_hot = torch.zeros_like(cosine)
                    #     one_hot.scatter_(1, class_num.view(-1, 1), 1.0)
                    #     logits = (cosine - one_hot * cos_margin) * cos_scale
                    #     # 6. 交叉熵损失（与前面 cal_type==4 用同一个 self.criterion）
                    #     loss_embed = self.criterion(logits, class_num)
                    #     lcls += loss_embed
                    #
                    # if cal_type == 6:  #2190类15维ElasticFace
                    #     # ElasticFace-Cos 超参数
                    #     cos_scale = 30.0  # 缩放因子 s
                    #     mean_m = 0.35  # 高斯分布均值（原 CosFace 的 margin）
                    #     sigma = 0.025  # 高斯分布标准差（弹性程度）
                    #     # 1. 取分类头前15维作为原始嵌入
                    #     raw_embed = pcls[:, :self.embed_dim].float()  # [n, 15]
                    #     # 2. BN + L2 归一化
                    #     # normed_embed = self.bn_embed(raw_embed)
                    #     normed_embed = F.normalize(raw_embed, p=2, dim=1)
                    #     # 3. 归一化类中心矩阵
                    #     w = F.normalize(self.cosface_weight, p=2, dim=1)  # [100, 15]
                    #     # 4. 计算余弦相似度
                    #     cosine = F.linear(normed_embed, w)  # [n, 100]
                    #     # 5. 为每个样本采样随机 margin
                    #     class_num = (tcls[i] - 1).long()  # 转为0-based标签
                    #     random_margin = torch.normal(mean=mean_m, std=sigma, size=(cosine.size(0),),
                    #                                  device=cosine.device)
                    #     random_margin = torch.clamp(random_margin, min=0.0)  # margin 非负
                    #     # 6. one-hot 掩码
                    #     one_hot = torch.zeros_like(cosine)
                    #     one_hot.scatter_(1, class_num.view(-1, 1), 1.0)
                    #     # 7. 在真实类上减去随机 margin
                    #     cosine_m = cosine - one_hot * random_margin.unsqueeze(1)
                    #     cosine_m = torch.clamp(cosine_m, -1.0 + 1e-7, 1.0 - 1e-7)  # 数值稳定
                    #     # 8. 缩放 + 交叉熵损失
                    #     logits = cosine_m * cos_scale
                    #     loss_embed = self.criterion(logits, class_num)
                    #     lcls += loss_embed


            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp["box"]
        lobj *= self.hyp["obj"]*5
        lcls *= self.hyp["cls"]
        bs = tobj.shape[0]  # batch size

        return (lbox + lobj + lcls) * bs, torch.cat((lbox, lobj, lcls)).detach()

    def bilinear_interpolation(img, coords):
        """
        对给定坐标进行双线性插值
        :param img: 输入的图像（二维）
        :param coords: 坐标，形状为 [N, 2] 的张量
        :return: 插值后的灰度值
        """
        # 计算整数坐标和插值比率
        x = coords[:, 0]
        y = coords[:, 1]

        x0 = torch.floor(x).long()
        y0 = torch.floor(y).long()
        x1 = x0 + 1
        y1 = y0 + 1

        # 确保坐标在图像边界内
        x0 = torch.clamp(x0, 0, img.shape[1] - 1)
        y0 = torch.clamp(y0, 0, img.shape[0] - 1)
        x1 = torch.clamp(x1, 0, img.shape[1] - 1)
        y1 = torch.clamp(y1, 0, img.shape[0] - 1)

        # 获取四个邻近点的值
        Ia = img[y0, x0]
        Ib = img[y0, x1]
        Ic = img[y1, x0]
        Id = img[y1, x1]

        # 计算权重
        wa = (x1 - x) * (y1 - y)
        wb = (x - x0) * (y1 - y)
        wc = (x1 - x) * (y - y0)
        wd = (x - x0) * (y - y0)

        # 使用权重进行插值
        return wa * Ia + wb * Ib + wc * Ic + wd * Id
    def build_targets(self, p, targets):
        """Prepares model targets from input targets (image,class,x,y,w,h) for loss computation, returning class, box,
        indices, and anchors.
        """
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        tcls, tbox, indices, anch, tcls_add = [], [], [], [], []
        gain = torch.ones(7, device=self.device)  # zxj normalized to grid space gain
        ai = torch.arange(na, device=self.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[..., None]), 2)  # append anchor indices

        g = 0.5  # bias
        off = (
            torch.tensor(
                [
                    [0, 0],
                    [1, 0],
                    [0, 1],
                    [-1, 0],
                    [0, -1],  # j,k,l,m
                    # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                ],
                device=self.device,
            ).float()
            * g
        )  # offsets

        for i in range(self.nl):
            anchors, shape = self.anchors[i], p[i].shape
            gain[2:6] = torch.tensor(shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            t = targets * gain  # shape(3,n,7)
            if nt:
                # Matches
                r = t[..., 4:6] / anchors[:, None]  # wh ratio
                j = torch.max(r, 1 / r).max(2)[0] < self.hyp["anchor_t"]  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # filter


                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1 < g) & (gxy > 1)).T
                l, m = ((gxi % 1 < g) & (gxi > 1)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            c_add=[]
            # if self.nc>15:
            #     bc=t[:,:2]
            #     gxy=t[:,2:4]
            #     gwh = t[:, 4:6]
            #     c_add = t[:, 6:10]
            #     a = t[:, 10:]
            # else:
            bc, gxy, gwh, a = t.chunk(4, 1)  #zXj (image, class), grid xy, grid wh, anchors
            a, (b, c) = a.long().view(-1), bc.long().T  # anchors, image, class
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid indices

            # Append
            indices.append((b, a, gj.clamp_(0, shape[2] - 1), gi.clamp_(0, shape[3] - 1)))  # image, anchor, grid
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class
            tcls_add.append(c_add)

        return tcls, tbox, indices, anch, tcls_add
