import numpy as np
import cv2

# Nhiễu muối tiêu
def add_sp_noise(image, salt_rate = 0.04, pepper_rate = 0.06):
    mask = np.random.choice(a = [0, 1, 2], size = (image.shape[0], image.shape[1]), replace = True, p = [salt_rate, 1 - salt_rate - pepper_rate, pepper_rate])
    new_image = image.copy()
    new_image[mask == 0] = 255
    new_image[mask == 2] = 0
    return new_image

# Nhiễu Gaussian
def add_gausian_noise(image, mean = 0, std = 15):
    new_image = image.copy().astype(np.float64)
    noise = np.random.normal(mean, std, size = image.shape)
    new_image = new_image + noise
    new_image = np.clip(new_image, 0, 255)
    return new_image.astype(np.uint8)