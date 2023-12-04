import os
from PIL import Image, ImageFilter

################################################################################
################### MEDIAN FILTER ############################################
def apply_median_filter(image):
    # Open the image file
    # original_image = Image.open(image_path)

    # Apply median filter
    filtered_image = image.filter(ImageFilter.MedianFilter())
    return filtered_image

    # Create the output path
    # filename = os.path.basename(image_path)
    # output_image_path = os.path.join(output_folder, filename)

    # Save the filtered image
    # filtered_image.save(output_image_path)

# def apply_median_filter_to_images(input_folder, output_folder):
#     # Create the output folder if it doesn't exist
#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder)

#     # Iterate over each file in the input folder
#     for filename in os.listdir(input_folder):
#         if filename.endswith(".jpg") or filename.endswith(".png"):
#             # Read the image
#             input_image_path = os.path.join(input_folder, filename)

#             # Apply median filter
#             apply_median_filter(input_image_path, output_folder)


# input_folder_path = '/content/images1' # You can change the path accordingly - just the path of the folder with images
# output_folder_path = '/content/images1' # You can change the path accordingly - just the path of the folder with images

# apply_median_filter_to_images(input_folder_path, output_folder_path)
#################################################################################