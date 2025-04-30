import numpy as np
import matplotlib.pyplot as plt

filters = ["Sepia", "Grayscale", "Posterize", "Negative", "Solarize"]

print("Which filter do you want to apply? ")
for index, filter in enumerate(filters):
    print(f"{index + 1}. {filter}")

filter_index = input("Enter the index: ")

try:
    image_array = np.load("original_image.npy")
    R = image_array[..., 0]
    G = image_array[..., 1]
    B = image_array[..., 2]

    plt.figure(figsize=(10, 6))
    plt.subplot(121)
    to_show_image_array = image_array.astype(np.uint8)
    plt.imshow(to_show_image_array)
    plt.title("Original Image")
    plt.grid(False)

    if filter_index == "1":
        sepia_R = 0.393 * R + 0.769 * G + 0.189 * B
        sepia_G = 0.349 * R + 0.686 * G + 0.168 * B
        sepia_B = 0.272 * R + 0.534 * G + 0.131 * B

        sepia_filtered = np.stack([sepia_R, sepia_G, sepia_B], axis=-1)
        sepia_image = np.clip(sepia_filtered, 0, 255).astype(np.uint8)

        plt.subplot(122)
        plt.imshow(sepia_image)
        plt.title("Sepia Filter")
        plt.grid(False)
        plt.show()

    elif filter_index == "2":
        gray = 0.299 * R + 0.587 * G + 0.114 * B
        gray_image = np.stack([gray, gray, gray], axis=-1).astype(np.uint8)

        plt.subplot(122)
        plt.imshow(gray_image)
        plt.title("Grayscale Filter")
        plt.grid(False)
        plt.show()

    elif filter_index == "3":
        levels = 9
        step = 256 // levels

        posterized = (image_array // step) * step
        posterized_image = np.clip(posterized, 0, 255).astype(np.uint8)
        plt.subplot(122)
        plt.imshow(posterized_image)
        plt.title("Posterized Filter")
        plt.grid(False)
        plt.show()

    elif filter_index == "4":
        negative_image = 255 - image_array.astype(np.uint8)

        plt.subplot(122)
        plt.imshow(negative_image)
        plt.title("Negative Filter")
        plt.grid(False)
        plt.show()

    elif filter_index == "5":
        threshold = 128
        solarized_image = np.where(image_array > threshold, 255 - image_array, image_array).astype(np.uint8)

        plt.subplot(122)
        plt.imshow(solarized_image)
        plt.title("Solarized Filter")
        plt.grid(False)
        plt.show()

    else:
        print("Incorrect index!")

except Exception as e:
    print("Error processing image: ", e)
