# -*- coding: utf-8 -*-
# Copyright (c) XiMing Xing. All rights reserved.
# Author: XiMing Xing
# Description:
import torch
import torch.nn as nn
import torchvision
import numpy as np
from torchvision.transforms import ToPILImage
import os



def channel_saturation_penalty_loss(x: torch.Tensor):
    assert x.shape[1] == 3
    r_channel = x[:, 0, :, :]
    g_channel = x[:, 1, :, :]
    b_channel = x[:, 2, :, :]
    channel_accumulate = torch.pow(r_channel, 2) + torch.pow(g_channel, 2) + torch.pow(b_channel, 2)
    return channel_accumulate.mean() / 3


def area(a, b, c):
    return (c[1] - a[1]) * (b[0] - a[0]) - (b[1] - a[1]) * (c[0] - a[0])


def triangle_area(A, B, C):
    out = (C - A).flip([-1]) * (B - A)
    out = out[..., 1] - out[..., 0]
    return out


def compute_sine_theta(s1, s2):  # s1 and s2 aret two segments to be uswed
    # s1, s2 (2, 2)
    v1 = s1[1, :] - s1[0, :]
    v2 = s2[1, :] - s2[0, :]
    # print(v1, v2)
    sine_theta = (v1[0] * v2[1] - v1[1] * v2[0]) / (torch.norm(v1) * torch.norm(v2))
    return sine_theta


def xing_loss_fn(x_list, scale=1e-3):  # x[npoints, 2]
    loss = 0.
    # print(f"points_len: {len(x_list)}")
    for x in x_list:
        # print(f"x: {x}")
        seg_loss = 0.
        N = x.size()[0]
        assert N % 3 == 0, f'The segment number ({N}) is not correct!'
        x = torch.cat([x, x[0, :].unsqueeze(0)], dim=0)  # (N+1,2)
        segments = torch.cat([x[:-1, :].unsqueeze(1), x[1:, :].unsqueeze(1)], dim=1)  # (N, start/end, 2)
        segment_num = int(N / 3)
        for i in range(segment_num):
            cs1 = segments[i * 3, :, :]  # start control segs
            cs2 = segments[i * 3 + 1, :, :]  # middle control segs
            cs3 = segments[i * 3 + 2, :, :]  # end control segs
            # print('the direction of the vectors:')
            # print(compute_sine_theta(cs1, cs2))
            direct = (compute_sine_theta(cs1, cs2) >= 0).float()
            opst = 1 - direct  # another direction
            sina = compute_sine_theta(cs1, cs3)  # the angle between cs1 and cs3
            seg_loss += direct * torch.relu(- sina) + opst * torch.relu(sina)
            # print(direct, opst, sina)
        seg_loss /= segment_num

        templ = seg_loss
        loss += templ * scale  # area_loss * scale

    return loss / (len(x_list))



class ToneLoss(nn.Module):
    def __init__(self, dist_loss_weight, pixel_dist_kernel_blur, pixel_dist_sigma):
        super(ToneLoss, self).__init__()
        self.dist_loss_weight = dist_loss_weight
        self.im_init = None
        self.mse_loss = nn.MSELoss()
        self.blurrer = torchvision.transforms.GaussianBlur(kernel_size=(pixel_dist_kernel_blur,
                                                                        pixel_dist_kernel_blur), sigma=(pixel_dist_sigma))

    def get_scheduler(self, step=None):
        if step is not None:
            return self.dist_loss_weight * np.exp(-(1/5)*((step-300)/(20)) ** 2)
        else:
            return self.dist_loss_weight

    def forward(self, let_raster, cur_raster, step=None):
        blurred_cur = self.blurrer(cur_raster)
        blurred_let = self.blurrer(let_raster)
        #self.save_tensor_as_png(blurred_cur, "out_cur")
        #self.save_tensor_as_png(blurred_let, "out_let")
        return self.mse_loss(blurred_let.detach(), blurred_cur) * self.get_scheduler(step)
    
    def save_tensor_as_png(self, tensor, output_dir='output', prefix='image'):
        to_pil = ToPILImage()

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for i in range(tensor.size(0)):
            img_tensor = tensor[i]
            img = to_pil(img_tensor)
            img.save(os.path.join(output_dir, f'{prefix}_{i}.png'), 'PNG')

        print(f"Images saved in '{output_dir}' as PNG files.")

class ImageToSDFLoss(nn.Module):
    def __init__(self, reduction='mean', threshold=0.5):
        super(ImageToSDFLoss, self).__init__()
        self.reduction = reduction
        self.threshold = threshold

    def forward(self, pred_img, target_img):
        """
        计算两个图像张量之间的SDF相似度损失

        参数:
            pred_img: 预测的图像张量 [B, C, H, W]
            target_img: 目标图像张量 [B, C, H, W]
        """
        # 1. 将图像转换为二值图像（如果不是已经二值化的）
        if pred_img.size(1) > 1:  # 多通道图像（如RGB）
            pred_gray = 0.299 * pred_img[:, 0] + 0.587 * pred_img[:, 1] + 0.114 * pred_img[:, 2]
            pred_binary = (pred_gray > self.threshold).float().unsqueeze(1)
        else:  # 单通道图像
            pred_binary = (pred_img > self.threshold).float()

        if target_img.size(1) > 1:  # 多通道图像
            target_gray = 0.299 * target_img[:, 0] + 0.587 * target_img[:, 1] + 0.114 * target_img[:, 2]
            target_binary = (target_gray > self.threshold).float().unsqueeze(1)
        else:  # 单通道图像
            target_binary = (target_img > self.threshold).float()

        # 2. 计算二值图像的SDF (通过可微分的近似方法)
        pred_sdf = self._compute_sdf(pred_binary)
        target_sdf = self._compute_sdf(target_binary)

        # 3. 计算SDF之间的相似度损失
        # L1损失
        l1_loss = F.l1_loss(pred_sdf, target_sdf, reduction='none')

        # 梯度一致性损失
        grad_pred = self._compute_gradients(pred_sdf)
        grad_target = self._compute_gradients(target_sdf)

        # 添加小常数避免除零
        grad_norm_pred = torch.norm(grad_pred, dim=1, keepdim=True) + 1e-6
        grad_norm_target = torch.norm(grad_target, dim=1, keepdim=True) + 1e-6

        # 归一化梯度
        grad_pred_norm = grad_pred / grad_norm_pred
        grad_target_norm = grad_target / grad_norm_target

        # 计算余弦相似度
        cos_sim = (grad_pred_norm * grad_target_norm).sum(dim=1)
        grad_loss = (1 - cos_sim).mean(dim=[1, 2])

        # 零水平集损失
        zero_level_mask = (torch.abs(target_sdf) < 0.05).float()
        zero_level_loss = (l1_loss * zero_level_mask).sum(dim=[1, 2, 3]) / (zero_level_mask.sum(dim=[1, 2, 3]) + 1e-6)

        # 组合损失
        total_loss = l1_loss.mean(dim=[1, 2, 3]) + 0.5 * grad_loss + 0.5 * zero_level_loss

        if self.reduction == 'mean':
            return total_loss.mean()
        elif self.reduction == 'sum':
            return total_loss.sum()
        else:
            return total_loss

    def _compute_sdf(self, binary_img):
        """
        计算二值图像的近似SDF
        使用距离变换的可微分近似
        """
        batch_size = binary_img.shape[0]

        fg_dist = self._distance_transform(binary_img)
        sdf = fg_dist

        return sdf

    def _distance_transform(self, binary_img):
        """
        二值图像的可微分距离变换近似
        使用高斯模糊作为近似
        """
        # 使用多尺度高斯模糊来近似距离变换
        blurred = binary_img
        distances = torch.zeros_like(binary_img)

        # 使用不同核大小的高斯模糊
        sigma_values = [1, 2, 4, 8, 16]
        for sigma in sigma_values:
            kernel_size = 2 * int(3 * sigma) + 1  # 确保核大小足够大
            padding = kernel_size // 2

            # 应用高斯模糊
            blurred = F.gaussian_blur(binary_img, kernel_size=[kernel_size, kernel_size],
                                      sigma=[sigma, sigma])

            # 累积距离 (1-值表示距离)
            mask = (blurred < 0.99) & (blurred > 0.01)
            contribution = mask.float() * (1 - blurred) * sigma
            distances = distances + contribution

        return distances

    def _compute_gradients(self, tensor):
        """计算张量的空间梯度"""
        batch_size = tensor.shape[0]

        # 使用Sobel滤波器计算梯度
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3).to(
            tensor.device)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3).to(
            tensor.device)

        dx = F.conv2d(tensor, sobel_x.expand(tensor.size(1), -1, -1, -1), padding=1, groups=tensor.size(1))
        dy = F.conv2d(tensor, sobel_y.expand(tensor.size(1), -1, -1, -1), padding=1, groups=tensor.size(1))

        return torch.cat([dx, dy], dim=1)
