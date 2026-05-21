import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F



def compute_sdf(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("无法加载图像，请检查路径是否正确。")
    _, binary_image = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    binary_image = binary_image // 255
    external_distance = cv2.distanceTransform(1 - binary_image, cv2.DIST_L2, 5)
    internal_distance = cv2.distanceTransform(binary_image, cv2.DIST_L2, 5)
    sdf = external_distance - internal_distance
    return sdf


def visualize_sdf_with_threshold(sdf, threshold_up=100, threshold_low=2, output_path=None):

    sdf_normalized = (sdf - sdf.min()) / (sdf.max() - sdf.min()) * 255
    sdf_normalized = sdf_normalized.astype(np.uint8)

    sdf_normalized[sdf_normalized > threshold_up] = 255
    sdf_normalized[sdf_normalized < threshold_low] = 0

    if output_path:
        plt.imsave(output_path, sdf_normalized, cmap='gray')
        print(f"SDF 灰度图已保存到：{output_path}")
    else:
        plt.show()


if __name__ == "__main__":
    # 输入 PNG 图像路径
    image_path = 'R.png'   # 替换为实际路径
    # 计算 SDF
    sdf_result = compute_sdf(image_path)
    # 可视化 SDF
    visualize_sdf_with_threshold(sdf_result, threshold_up=80, threshold_low=20, output_path="int_r.png")