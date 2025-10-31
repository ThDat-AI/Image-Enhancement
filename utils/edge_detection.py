import numpy as np
import cv2

# Sobel
def edge_Sobel(image, ksize = 3):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    grad_x = cv2.Sobel(gray, ddepth = cv2.CV_64F, dx = 1, dy = 0, ksize = ksize)
    grad_y = cv2.Sobel(gray, ddepth = cv2.CV_64F, dx = 0, dy = 1, ksize = ksize)

    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)

    grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

    return grad

# Prewitt
def edge_Prewitt(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    kernelx = np.array([[-1, 0, 1],
                       [-1, 0, 1],
                       [-1, 0, 1]], dtype=np.float32)
    
    kernely = np.array([[1, 1, 1],
                        [0, 0, 0],
                        [-1, -1, -1]], dtype=np.float32)
    
    grad_x = cv2.filter2D(gray, ddepth = cv2.CV_64F, kernel = kernelx)
    grad_y = cv2.filter2D(gray, ddepth = cv2.CV_64F, kernel = kernely)

    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)

    grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

    return grad

# Canny
def edge_Canny(image, low_threshold = 100, high_threshold = 200):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    edges = cv2.Canny(gray, threshold1 = low_threshold, threshold2 = high_threshold)

    return edges