import os
from PIL import Image, ImageFilter
import numpy as np
from scipy.ndimage import convolve

################################################################################
################### MEAN FILTER ############################################
def apply_mean_filter(image, kernel_size):
    # Open the image file
    # original_image = Image.open(image_path)

    # Convert the image to a NumPy array
    image_array = np.array(image)

    # Apply mean filter
    filtered_image_array = apply_mean_filter_numpy(image_array, kernel_size)

    # Convert the filtered NumPy array back to an image
    filtered_image = Image.fromarray(filtered_image_array)

    # Create the output path
    # filename = os.path.basename(image_path)
    # output_image_path = os.path.join(output_folder, filename)

    # Save the filtered image
    # filtered_image.save(output_image_path)
    return filtered_image

def apply_mean_filter_numpy(image_array, kernel_size):
    if len(image_array.shape) == 3:  # Color image
        # Pad the image array to handle pixels at the borders
        pad_width = kernel_size // 2
        padded_image = np.pad(image_array, pad_width=((pad_width, pad_width), (pad_width, pad_width), (0, 0)), mode='constant')

        # Define the mean filter kernel
        kernel = np.ones((kernel_size, kernel_size), dtype=np.float32) / (kernel_size ** 2)

        # Apply the mean filter separately for each channel
        filtered_image = np.zeros_like(image_array, dtype=np.float32)
        for c in range(image_array.shape[2]):
            filtered_image[:, :, c] = convolve(padded_image[:, :, c], kernel, mode='constant')[1:-1, 1:-1]

    else:  # Grayscale image
        # Pad the image array to handle pixels at the borders
        pad_width = kernel_size // 2
        padded_image = np.pad(image_array, pad_width=((pad_width, pad_width), (pad_width, pad_width)), mode='constant')

        # Define the mean filter kernel
        kernel = np.ones((kernel_size, kernel_size), dtype=np.float32) / (kernel_size ** 2)

        # Apply the mean filter
        filtered_image = convolve(padded_image, kernel, mode='constant')[1:-1, 1:-1]

    return filtered_image.astype(np.uint8)

# def apply_mean_filter_to_images(input_folder, output_folder, kernel_size):
#     # Create the output folder if it doesn't exist
#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder)

#     # Iterate over each file in the input folder
#     for filename in os.listdir(input_folder):
#         if filename.endswith(".jpg") or filename.endswith(".png"):
#             # Read the image
#             input_image_path = os.path.join(input_folder, filename)

#             # Apply mean filter
#             apply_mean_filter(input_image_path, output_folder, kernel_size)


# input_folder_path = '/content/images1' # You can change the path accordingly - just the path of the folder with images
# output_folder_path = '/content/images1' # You can change the path accordingly - just the path of the folder with images
# kernel_size = 3

# apply_mean_filter_to_images(input_folder_path, output_folder_path, kernel_size)
#################################################################################