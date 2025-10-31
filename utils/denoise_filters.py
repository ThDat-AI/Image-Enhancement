import numpy as np
import cv2

# Dùng mean
def denoise_mean(image, ksize = 3):
    kernel = np.ones((ksize, ksize)) * (1 / ksize**2)
    new_image = cv2.filter2D(image, ddepth = -1, kernel = kernel)
    return new_image

# Dùng median
def denoise_median(image, ksize = 3):
    new_image = cv2.medianBlur(image, ksize = ksize)
    return new_image

# Dùng Gaussian blur
def denoise_gaussian(image, ksize = 3, sigmaX = 0, sigmaY = 0):
    new_image = cv2.GaussianBlur(image, ksize = (ksize, ksize), sigmaX = sigmaX, sigmaY= sigmaY)
    return new_image

def blur_image(image, ksize = 5):
    # Ensure ksize is odd for proper kernel behavior in some cases,
    # though cv2.blur can handle even. Let's make it odd for consistency with other filters.
    if ksize % 2 == 0:
        ksize += 1
    new_image = cv2.blur(image, ksize = (ksize, ksize))
    return new_image