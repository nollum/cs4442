import cv2
import numpy as np
import matplotlib.pyplot as plt

def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def thresholded(center, pixels):
    thresholded_pixels = []
    for a_pixel in pixels:
        if a_pixel >= center:
            thresholded_pixels.append(1)
        else:
            thresholded_pixels.append(0)
    return thresholded_pixels

def get_pixel_values(image, x, y):
    pixels = []
    pixels.append(image[x - 1, y - 1])
    pixels.append(image[x - 1, y])
    pixels.append(image[x - 1, y + 1])
    pixels.append(image[x, y + 1])
    pixels.append(image[x + 1, y + 1])
    pixels.append(image[x + 1, y])
    pixels.append(image[x + 1, y - 1])
    pixels.append(image[x, y - 1])
    return pixels

def compute_lbp(image):
    height, width = image.shape
    lbp_image = np.zeros((height-2, width-2), dtype=np.uint8)

    for i in range(1, height-1):
        for j in range(1, width-1):
            center = image[i, j]
            pixels = get_pixel_values(image, i, j)
            values = thresholded(center, pixels)
            weights = [1, 2, 4, 8, 16, 32, 64, 128]
            lbp_value = sum([v * w for v, w in zip(values, weights)])
            lbp_image[i-1, j-1] = lbp_value

    return lbp_image

def display_lbp(image, lbp):
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')

    plt.subplot(1, 2, 2)
    plt.imshow(lbp, cmap='gray')
    plt.title('LBP')

    plt.show()

def lbp_feature_extraction(image):
    gray_image = grayscale(image)
    lbp_image = compute_lbp(gray_image)
    flattened_lbp = lbp_image.flatten()
    
    return flattened_lbp

# print(lbp_feature_extraction('./train_filtered/vegetarian/1_Image_10.jpg').shape)
# image_path = '124_Image_81.jpeg'
# image = cv2.imread(image_path)
# gray_image = grayscale(image)

# lbp_image = compute_lbp(gray_image)
# display_lbp(gray_image, lbp_image)

# flattened_lbp = lbp_image.flatten()
# plt.hist(flattened_lbp, bins=256, range=[0, 256], density=True, cumulative=False)
# plt.title('LBP Histogram')
# plt.xlabel('LBP Value')
# plt.ylabel('Frequency')
# plt.show()
