from pathlib import Path
from sklearn import svm, metrics
from sklearn.utils import Bunch
from sklearn.model_selection import train_test_split
from skimage.io import imread
from skimage.transform import resize
from skimage import exposure
from skimage.color import rgb2hsv
import numpy as np

def compute_histogram(image):
    hsv_image = rgb2hsv(image)
    # experiment with bin numbers here
    hist_hue, _ = np.histogram(hsv_image[:, :, 0], bins=50, range=[0, 1])
    hist_saturation, _ = np.histogram(hsv_image[:, :, 1], bins=50, range=[0, 1])
    hist_value, _ = np.histogram(hsv_image[:, :, 2], bins=50, range=[0, 1])
    histogram_features = np.concatenate((hist_hue, hist_saturation, hist_value))
    return histogram_features

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
        img_resized = resize(img, (64, 64), anti_aliasing=True, mode='reflect')
        hist_features = compute_histogram(img_resized)
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
