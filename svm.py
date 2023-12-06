from pathlib import Path
from sklearn import svm, metrics
from sklearn.utils import Bunch
from sklearn.model_selection import train_test_split
from skimage.io import imread
from skimage.color import rgb2hsv
import numpy as np
from color_hist import color_hist_feature_extraction
from lbp import lbp_feature_extraction
from hog import hog_feature_extraction
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

descr = "An image classification dataset"
features = []
target = []

container_path = './data_filtered'
image_dir = Path(container_path)
folders = [directory for directory in image_dir.iterdir() if directory.is_dir()]
categories = [fo.name for fo in folders]

for i, direc in enumerate(folders):
    for file in direc.iterdir():
        img = imread(file)
        hist_features = color_hist_feature_extraction(img)
        lbp_features = lbp_feature_extraction(img)
        hog_features = hog_feature_extraction(img)
        combined_features = np.concatenate((hist_features, lbp_features, hog_features))
        features.append(combined_features)
        target.append(i)

features = np.array(features)
target = np.array(target)

scaler = StandardScaler()
features_standardized = scaler.fit_transform(features)

image_dataset = Bunch(data=features_standardized,
                      target=target,
                      target_names=categories,
                      DESCR=descr)

X_train, X_test, y_train, y_test = train_test_split(
    image_dataset.data, image_dataset.target, test_size=0.3, random_state=109)

clf = svm.SVC(kernel='linear')
cv_scores = cross_val_score(clf, image_dataset.data, image_dataset.target, cv=5)

print("Cross-Validation Scores:", cv_scores)
print("Average Accuracy:", np.mean(cv_scores))

# clf = svm.SVC(kernel='linear')
# clf.fit(X_train, y_train)
# y_pred = clf.predict(X_test)

# print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
