from skimage.io import imshow, imread
from skimage.feature import hog
from skimage import exposure

#Histogram of Oriented Gradients (HOG) feature extraction for a given image
def hog_feature_extraction(image):
    hogFeatureVector, hogImage = hog(image, 
                                    orientations=8, 
                                    pixels_per_cell = (8,8), 
                                    cells_per_block=(2,2),
                                    visualize=True,
                                    channel_axis=-1)

    #This is just for visualization, we only need to store hogFeatureVector
    # hogExampleImage = exposure.rescale_intensity(hogImage, in_range=(0,10))
    # imshow(hogExampleImage)

    #the feature of the image
    return hogFeatureVector


# image = imread("train_filtered/non-vegetarian")
# imshow(image)

#This is only an example
# hog_feature_extraction("train_filtered/non-vegetarian/0AG0XFSLIQXY.jpg")

