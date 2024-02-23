import cv2
from matplotlib import pyplot as plt
import numpy as np

def convolve(image, kernel):
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape

    conv_height = image_height - kernel_height + 1
    conv_width = image_width - kernel_width + 1
    conv_result = np.zeros((conv_height, conv_width), dtype=np.float32)

    for i in range(conv_height):
        for j in range(conv_width):
            roi = image[i:i+kernel_height, j:j+kernel_width]

            conv_result[i, j] = np.sum(np.multiply(roi, kernel))

    return conv_result

def gaussian_blur(img, kernel_size):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def sobel_operator(image, orientation):
    if orientation == 'x':
        kernel = np.array([[-1, 0, 1],
                           [-2, 0, 2],
                           [-1, 0, 1]])
    elif orientation == 'y':
        kernel = np.array([[-1, -2, -1],
                           [0, 0, 0],
                           [1, 2, 1]])
    else:
        raise ValueError("Orientation should be 'x' or 'y'")

    return convolve(image, kernel)

def compute_gradient_magnitude(dx, dy):
    return np.sqrt(dx**2 + dy**2)

def compute_gradient_direction(dx, dy):
    return np.arctan2(dy, dx)

def non_max_suppression(magnitude, direction):
    M, N = magnitude.shape
    suppressed = np.zeros((M, N), dtype=np.uint8)
    angle = direction * 180. / np.pi
    angle[angle < 0] += 180

    for i in range(1, M-1):
        for j in range(1, N-1):
            q = 255
            r = 255

            # Angle 0
            if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                q = magnitude[i, j+1]
                r = magnitude[i, j-1]
            # Angle 45
            elif 22.5 <= angle[i, j] < 67.5:
                q = magnitude[i+1, j-1]
                r = magnitude[i-1, j+1]
            # Angle 90
            elif 67.5 <= angle[i, j] < 112.5:
                q = magnitude[i+1, j]
                r = magnitude[i-1, j]
            # Angle 135
            elif 112.5 <= angle[i, j] < 157.5:
                q = magnitude[i-1, j-1]
                r = magnitude[i+1, j+1]

            if magnitude[i, j] >= q and magnitude[i, j] >= r:
                suppressed[i, j] = magnitude[i, j]
            else:
                suppressed[i, j] = 0

    return suppressed

def hysteresis_thresholding(img, low_threshold, high_threshold):
    M, N = img.shape
    weak = np.int32(25)
    strong = np.int32(255)

    strong_i, strong_j = np.where(img >= high_threshold)
    zeros_i, zeros_j = np.where(img < low_threshold)
    weak_i, weak_j = np.where((img <= high_threshold) & (img >= low_threshold))

    img_out = np.zeros((M, N), dtype=np.uint8)

    img_out[strong_i, strong_j] = strong
    img_out[weak_i, weak_j] = weak

    for i in range(1, M-1):
        for j in range(1, N-1):
            if img_out[i, j] == weak:
                try:
                    if (img_out[i+1, j-1] == strong) or (img_out[i+1, j] == strong) or (img_out[i+1, j+1] == strong) or \
                            (img_out[i, j-1] == strong) or (img_out[i, j+1] == strong) or \
                            (img_out[i-1, j-1] == strong) or (img_out[i-1, j] == strong) or (img_out[i-1, j+1] == strong):
                        img_out[i, j] = strong
                    else:
                        img_out[i, j] = 0
                except IndexError as e:
                    pass

    return img_out

def canny_edge_detection(image, low_threshold, high_threshold, kernel_size):
    blurred = gaussian_blur(image, kernel_size)
    dx = sobel_operator(blurred, 'x')
    dy = sobel_operator(blurred, 'y')
    magnitude = compute_gradient_magnitude(dx, dy)
    direction = compute_gradient_direction(dx, dy)
    non_max_suppressed = non_max_suppression(magnitude, direction)
    thresholded = hysteresis_thresholding(non_max_suppressed, low_threshold, high_threshold)
    return thresholded


image = cv2.imread('edge_detection.png', cv2.IMREAD_GRAYSCALE)
edges = canny_edge_detection(image, low_threshold=50, high_threshold=150, kernel_size=3)

plt.subplot(121),plt.imshow(image,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
plt.show()

