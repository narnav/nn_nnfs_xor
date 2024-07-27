import numpy as np
from PIL import Image

# Load the image
image = Image.open('sea.jpg')
image_np = np.array(image)

# Function to invert colors
def invert_colors(image):
    return 255 - image

# Function to convert to grayscale
def convert_to_grayscale(image):
    return np.dot(image[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)

# Function to adjust brightness
def adjust_brightness(image, factor):
    return np.clip(image + factor, 0, 255).astype(np.uint8)

# Invert colors
inverted_image_np = invert_colors(image_np)
inverted_image = Image.fromarray(inverted_image_np)
inverted_image.show()
inverted_image.save('path_to_save_inverted_image.jpg')

# Convert to grayscale
grayscale_image_np = convert_to_grayscale(image_np)
grayscale_image = Image.fromarray(grayscale_image_np)
grayscale_image.show()
grayscale_image.save('path_to_save_grayscale_image.jpg')

# Adjust brightness
brightened_image_np = adjust_brightness(image_np, 50)
brightened_image = Image.fromarray(brightened_image_np)
brightened_image.show()
brightened_image.save('path_to_save_brightened_image.jpg')

darkened_image_np = adjust_brightness(image_np, -50)
darkened_image = Image.fromarray(darkened_image_np)
darkened_image.show()
darkened_image.save('path_to_save_darkened_image.jpg')
