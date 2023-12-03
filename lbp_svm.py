from pathlib import Path
from sklearn import svm, metrics
from sklearn.utils import Bunch
from sklearn.model_selection import train_test_split
from skimage.io import imread
from skimage.transform import resize
from skimage import exposure
from skimage.color import rgb2hsv
import numpy as np
import cv2
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

    flattened_lbp = lbp_image.flatten()
    return flattened_lbp

descr = "An image classification dataset"
images = []
histogram_features = []
target = []

container_path = './'
image_dir = Path(container_path)
folders = [directory for directory in image_dir.iterdir() if directory.is_dir()]
categories = [fo.name for fo in folders]

for i, direc in enumerate(folders):
    for file in direc.iterdir():
        img = imread(file)
        image_gray = grayscale(img)
        img_resized = resize(image_gray, (64, 64), anti_aliasing=True, mode='reflect')
        hist_features = compute_lbp(img_resized)
        histogram_features.append(hist_features) 
        images.append(img_resized)
        target.append(i)

histogram_features = np.array(histogram_features)
target = np.array(target)
images = np.array(images)

image_dataset = Bunch(data=histogram_features,
                      target=target,
                      target_names=categories,
                      DESCR=descr)

X_train, X_test, y_train, y_test = train_test_split(
    image_dataset.data, image_dataset.target, test_size=0.3, random_state=109)

clf = svm.SVC(kernel='linear')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
