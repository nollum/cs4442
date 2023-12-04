import os
from PIL import Image

###############################################################################
########## RESIZE IMAGES ##########
def resize_image(input_path, new_size):
    # Open the image file
    original_image = Image.open(input_path)

    # Resize the image
    resized_image = original_image.resize(new_size)

    resized_image = resized_image.convert('RGB')

    # Save the resized image
    # resized_image.save(output_path)

    #Return
    return resized_image

# def resize_images_in_folder(input_folder, output_folder, new_size):
#     # Create the output folder if it doesn't exist
#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder)

#     # Iterate over each file in the input folder
#     for filename in os.listdir(input_folder):
#         if filename.endswith(".jpg") or filename.endswith(".png"):
#             # Read the image
#             input_image_path = os.path.join(input_folder, filename)

#             # Create the output path
#             output_image_path = os.path.join(output_folder, filename)

#             # Resize the image
#             resize_image(input_image_path, output_image_path, new_size)


# input_folder_path = '/content/images1'  # You can change the path accordingly - just the path of the folder with images
# output_folder_path = '/content/images1' # You can change the path accordingly - just the path of the folder with image
# new_size = (64, 64)

# resize_images_in_folder(input_folder_path, output_folder_path, new_size)
################################################################################