from image_resizing import resize_image
from gaussian_filter import apply_gaussian_filter
from mean_filter import apply_mean_filter
from median_filter import apply_median_filter
import os
import matplotlib.pyplot as plt


def filter_all_images(input_folder, output_folder):
    if(not os.path.isdir(output_folder)):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                # Read the image
                input_image_path = os.path.join(input_folder, filename)

                # Resize the image
                resizedImage = resize_image(input_image_path, (64,64))

                #Applying all the filters
                filteredImage = apply_mean_filter(resizedImage, 3) #Change Kernel if needed
                filteredImage = apply_gaussian_filter(filteredImage, 1) #Change sigma if needed
                filteredImage = apply_median_filter(filteredImage)
                output_image_path = os.path.join(output_folder, filename)

                filteredImage.save(output_image_path)

                # break #Use this to test everthing before filtering ALL images if needed


filter_all_images('train/non-vegetarian', 'train_filtered/non-vegetarian') #Change this if needed
filter_all_images('train/vegetarian', 'train_filtered/vegetarian') #Change this if needed