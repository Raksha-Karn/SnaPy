import numpy as np
from PIL import Image

original_image = Image.open('original.jpg').convert("RGBA")
original_image_array = np.array(original_image)
print(original_image_array.shape)

rgb_image_array = original_image_array[:, :, :3]
print(rgb_image_array.shape)

with open('original_image.npy', 'wb') as img_file:
    np.save(img_file, rgb_image_array)