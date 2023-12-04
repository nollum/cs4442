import os
from PIL import Image
from scipy.ndimage import gaussian_filter
import numpy as np
def apply_gaussian_filter(image, sigma):
    # Open the image file
    # original_image = Image.open(image_path)

    # Convert the image to a NumPy array
    image_array = np.array(image)

    # Apply Gaussian filter
    filtered_image_array = apply_gaussian_filter_numpy(image_array, sigma)

    # Convert the filtered NumPy array back to an image
    filtered_image = Image.fromarray(filtered_image_array)

    # Create the output path
    # filename = os.path.basename(image_path)
    # output_image_path = os.path.join(output_folder, filename)

    if filtered_image.mode == 'RGBA':
        filtered_image = filtered_image.convert('RGB')

    # Save the filtered image
    # filtered_image.save(output_image_path)

    return filtered_image


def apply_gaussian_filter_numpy(image_array, sigma):
    # Apply Gaussian filter separately for each channel
    filtered_image = np.zeros_like(image_array, dtype=np.float32)
    for c in range(image_array.shape[2]):
        filtered_image[:, :, c] = gaussian_filter(image_array[:, :, c], sigma)

    return filtered_image.astype(np.uint8)

# def apply_gaussian_filter_to_images(input_folder, output_folder, sigma):
#     # Create the output folder if it doesn't exist
#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder)

#     # Iterate over each file in the input folder
#     for filename in os.listdir(input_folder):
#         if filename.endswith(".jpg") or filename.endswith(".png"):
#             # Read the image
#             input_image_path = os.path.join(input_folder, filename)

#             # Apply Gaussian filter
#             apply_gaussian_filter(input_image_path, output_folder, sigma)


# input_folder_path = '/content/images1' # You can change the path accordingly - just the path of the folder with images
# output_folder_path = '/content/images1' # You can change the path accordingly - just the path of the folder with images
# sigma = 1.5

# apply_gaussian_filter_to_images(input_folder_path, output_folder_path, sigma)