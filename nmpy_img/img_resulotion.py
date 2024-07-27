import numpy as np
from PIL import Image

# Load the image
image = Image.open('sea.jpg')
image_np = np.array(image)

# Function to downsample the image
def downsample_image(image, factor):
    return image[::factor, ::factor]

# Downsample the image by a factor of 2 (reduce resolution by half)
downsampled_image_np = downsample_image(image_np, 2)

# Convert back to an image
downsampled_image = Image.fromarray(downsampled_image_np)
downsampled_image.show()
# Save the downsampled image
downsampled_image.save('downsampled_image.jpg')

# rotate with pillow
downsampled_image.rotate(90, expand=True)
downsampled_image.show()