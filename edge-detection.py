# ONLY USE THIS FEATURE IF NECESSARY
import os
import cv2
import numpy as np

###############################################################################
################ EDGE DETECTION ##############################################
def apply_canny_edge_detection(image_array, low_threshold, high_threshold):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)

    # Apply Gaussian blur to reduce noise and improve edge detection
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # Apply the Canny edge detector
    edges = cv2.Canny(blurred_image, low_threshold, high_threshold)

    return edges

def apply_edge_detection_to_images(input_folder, output_folder, low_threshold, high_threshold):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Iterate over each file in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            # Read the image
            input_image_path = os.path.join(input_folder, filename)

            # Create the output path
            output_image_path = os.path.join(output_folder, filename)

            # Read the image file
            original_image = cv2.imread(input_image_path)
            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

            # Apply Canny edge detection
            edges_array = apply_canny_edge_detection(original_image, low_threshold, high_threshold)

            # Convert the resulting NumPy array back to an image
            edges_image = Image.fromarray(edges_array)

            # Save the image with edges
            edges_image.save(output_image_path)

input_folder_path = '/content/images1'
output_folder_path = '/content/images1'
low_threshold = 50
high_threshold = 150

apply_edge_detection_to_images(input_folder_path, output_folder_path, low_threshold, high_threshold)
###############################################################################