import numpy as np
import cv2

# utils làm nét ảnh
def sharp_image(image):
    kernel = np.array([[-1, -1, -1],
                       [-1, 9, -1],
                       [-1, -1, -1]])
    new_image = cv2.filter2D(image, ddepth = -1, kernel = kernel)
    return new_image