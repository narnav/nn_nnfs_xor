import numpy as np
from PIL import Image
import math

# Load the image
image = Image.open('sea.jpg')
image_np = np.array(image)

def rotate_image(image, angle):
    # Get the dimensions of the image
    h, w = image.shape[:2]

    # Convert the angle from degrees to radians
    angle_rad = math.radians(angle)

    # Calculate the center of the image
    center_x, center_y = w // 2, h // 2

    # Create an output image with the same size and type as the input
    rotated_image = np.zeros_like(image)

    # Calculate the rotation matrix
    cos_theta = math.cos(angle_rad)
    sin_theta = math.sin(angle_rad)

    for i in range(h):
        for j in range(w):
            # Calculate the coordinates of the pixel relative to the center
            y = i - center_y
            x = j - center_x

            # Calculate the new coordinates after rotation
            new_x = int(center_x + x * cos_theta - y * sin_theta)
            new_y = int(center_y + x * sin_theta + y * cos_theta)

            # Check if the new coordinates are within the bounds of the image
            if 0 <= new_x < w and 0 <= new_y < h:
                rotated_image[new_y, new_x] = image[i, j]

    return rotated_image

# Rotate the image by 45 degrees
rotated_image_np = rotate_image(image_np, 90)

# Convert back to an image
rotated_image = Image.fromarray(rotated_image_np)

# Show the rotated image
rotated_image.show()

# Save the rotated image
rotated_image.save('nmpy_rotate.jpg')
