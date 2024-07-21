import cv2
from google.colab.patches import cv2_imshow
import numpy as np

# Load ảnh màu
bg1_img = cv2.imread(
    '/content/drive/MyDrive/1. AI ALL IN ONE/2. Module 2/2. Week 2/CN/GreenBackground.png', 1)
ob_img = cv2.imread(
    '/content/drive/MyDrive/1. AI ALL IN ONE/2. Module 2/2. Week 2/CN/Object.png', 1)
new_bg_img = cv2.imread(
    '/content/drive/MyDrive/1. AI ALL IN ONE/2. Module 2/2. Week 2/CN/NewBackground.jpg', 1)

# resize kích thước, đồng nhất input
IMAGE_SIZE = (678, 381)
bg1_img = cv2.resize(bg1_img, IMAGE_SIZE)
ob_img = cv2.resize(ob_img, IMAGE_SIZE)
new_bg_img = cv2.resize(new_bg_img, IMAGE_SIZE)

# xám hóa ảnh, ép kiểu uint 8


def compute_difference(bg_img, ob_img):
    difference_three_channel = cv2.absdiff(bg_img, ob_img)
    difference_single_channel = np.sum(difference_three_channel, axis=2) / 3.0
    difference_single_channel = difference_single_channel.astype('uint8')

    return difference_single_channel


difference_single_channel = compute_difference(bg1_img, ob_img)
cv2_imshow(difference_single_channel)

# Chuyển đổi thành ảnh nhị phân 0-255


def compute_binary_mask(difference_single_channel):
    difference_binary = np.where(difference_single_channel >= 10, 255, 0)
    difference_binary = np.stack((difference_binary,)*3, axis=-1)
    return difference_binary


binary_mask = compute_binary_mask(difference_single_channel)
cv2_imshow(binary_mask)

# replace background


def replace_back_ground(bg1_img, new_bg_img, ob_img):
    difference_single_channel = compute_difference(bg1_img, ob_img)
    binary_mask = compute_binary_mask(difference_single_channel)
    output = np.where(binary_mask == 255, ob_img, new_bg_img)
    return output


output = replace_back_ground(bg1_img, new_bg_img, ob_img)

cv2_imshow(output)
