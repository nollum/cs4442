import skimage.io
import skimage.color
import numpy as np

def color_hist_feature_extraction(image):
    # convert image to RGB if not already in that format
    if image.shape[-1] == 1:
        image_rgb = skimage.color.gray2rgb(image)
    else:
        image_rgb = image

    channels = image_rgb[:, :, :3]  # Extract RGB channels
    channel_names = ("Red", "Green", "Blue")

    features = []
    for (channel, name) in zip(channels.transpose((2, 0, 1)), channel_names):
        channel_values = (channel * 255).astype(np.uint8)
        hist = np.histogram(channel_values, bins=256, range=(0, 256))[0]
        features.extend(hist.flatten())

    return features
