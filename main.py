import numpy as np
import matplotlib.pyplot as plt


def apply_image_filter(image_array):
    filters = ["Sepia", "Grayscale", "Posterize", "Negative", "Solarize"]

    print("Which filter do you want to apply? ")
    for index, filter in enumerate(filters):
        print(f"{index + 1}. {filter}")

    filter_index = input("Enter the index: ")

    try:
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

def apply_blurring(image, kernel_size=5, sigma=1.0):
    x = np.linspace(-(kernel_size // 2), kernel_size // 2, kernel_size)
    kernel_1d = np.exp(-0.5 * (x / sigma) ** 2)
    kernel_1d = kernel_1d / np.sum(kernel_1d)
    kernel = np.outer(kernel_1d, kernel_1d)

    def convolve2d(channel, kernel):
        h, w = kernel.shape
        pad_h, pad_w = h // 2, w // 2
        padded = np.pad(channel, ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')
        output = np.zeros_like(channel, dtype=np.float64)

        for i in range(channel.shape[0]):
            for j in range(channel.shape[1]):
                output[i, j] = np.sum(padded[i:i+h, j:j+w] * kernel)
        return output

    if len(image.shape) == 3:
        result = np.zeros_like(image, dtype=np.float64)
        for c in range(3):
            result[..., c] = convolve2d(image[..., c], kernel)
    else:
        result = convolve2d(image, kernel)

    plt.figure(figsize=(10, 6))
    plt.subplot(121)
    plt.imshow(image.astype(np.uint8))
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(122)
    plt.imshow(np.clip(result, 0, 255).astype(np.uint8))
    plt.title("Blurred Image")
    plt.grid(False)
    plt.show()

    return np.clip(result, 0, 255).astype(np.uint8)

def apply_sharpening(image_array):
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])

    def convolve2d(channel, kernel):
        k_h, k_w = kernel.shape
        pad_h, pad_w = k_h // 2, k_w // 2
        padded = np.pad(channel, ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')
        result = np.zeros_like(channel, dtype=np.float64)

        for i in range(channel.shape[0]):
            for j in range(channel.shape[1]):
                result[i, j] = np.sum(padded[i:i+k_h, j:j+k_w] * kernel)
        return result

    if len(image_array.shape) == 3:
        sharpened = np.zeros_like(image_array, dtype=np.float64)
        for c in range(3):
            sharpened[..., c] = convolve2d(image_array[..., c], kernel)
    else:
        sharpened = convolve2d(image_array, kernel)

    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)

    plt.figure(figsize=(10, 6))
    plt.subplot(121)
    plt.imshow(image_array.astype(np.uint8))
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(122)
    plt.imshow(sharpened)
    plt.title("Sharpened Image")
    plt.axis("off")
    plt.show()

def oil_painting_effect(image_array, radius=3, levels=20):
    if len(image_array.shape) == 3:
        gray = (0.299 * image_array[..., 0] +
                0.587 * image_array[..., 1] +
                0.114 * image_array[..., 2]).astype(np.uint8)
    else:
        gray = image_array.copy()

    h, w = gray.shape
    output = np.zeros_like(image_array)

    for i in range(radius, h - radius):
        for j in range(radius, w - radius):
            intensity_count = np.zeros(levels)
            r_sum = np.zeros(levels)
            g_sum = np.zeros(levels)
            b_sum = np.zeros(levels)

            for m in range(-radius, radius + 1):
                for n in range(-radius, radius + 1):
                    ii = i + m
                    jj = j + n
                    intensity = gray[ii, jj] * levels // 256
                    intensity = min(intensity, levels - 1)
                    intensity_count[intensity] += 1
                    r_sum[intensity] += image_array[ii, jj, 0]
                    g_sum[intensity] += image_array[ii, jj, 1]
                    b_sum[intensity] += image_array[ii, jj, 2]

            max_intensity = np.argmax(intensity_count)
            count = intensity_count[max_intensity] or 1

            output[i, j, 0] = r_sum[max_intensity] / count
            output[i, j, 1] = g_sum[max_intensity] / count
            output[i, j, 2] = b_sum[max_intensity] / count

    return output.astype(np.uint8)

def cartoon_effect(image_array, edge_threshold=80, posterize_levels=6):
    gray = 0.299 * image_array[..., 0] + 0.587 * image_array[..., 1] + 0.114 * image_array[..., 2]

    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1],
                        [0,  0,  0],
                        [1,  2,  1]])

    def convolve(channel, kernel):
        pad = kernel.shape[0] // 2
        padded = np.pad(channel, pad, mode='edge')
        result = np.zeros_like(channel)
        for i in range(channel.shape[0]):
            for j in range(channel.shape[1]):
                region = padded[i:i+kernel.shape[0], j:j+kernel.shape[1]]
                result[i, j] = np.sum(region * kernel)
        return result

    gx = convolve(gray, sobel_x)
    gy = convolve(gray, sobel_y)
    magnitude = np.sqrt(gx**2 + gy**2)
    edges = magnitude > edge_threshold

    step = 256 // posterize_levels
    posterized = (image_array // step) * step
    posterized = np.clip(posterized, 0, 255).astype(np.uint8)

    cartoon = posterized.copy()
    cartoon[edges] = 0

    return cartoon

def apply_image_effects(image_array):
    effects = ["Brightness & Contrast", "Oil Painting", "Cartoon"]
    for i, e in enumerate(effects):
        print(f"{i + 1}. {e}")

    choice = input("Choose effect index: ")

    if choice == "1":
        brightness = int(input("Enter brightness (-100 to 100): "))
        contrast = float(input("Enter contrast (e.g. 1.0 = no change): "))

        adjusted = (image_array.astype(np.float64) - 127.5) * contrast + 127.5 + brightness
        adjusted = np.clip(adjusted, 0, 255).astype(np.uint8)

        plt.imshow(adjusted)
        plt.title("Brightness & Contrast")
        plt.axis("off")
        plt.show()

    elif choice == "2":
        oil_img = oil_painting_effect(image_array)
        plt.imshow(oil_img)
        plt.title("Oil Painting Effect")
        plt.axis("off")
        plt.show()

    elif choice == "3":
        cartoon_img = cartoon_effect(image_array)
        plt.imshow(cartoon_img)
        plt.title("Cartoon Effect")
        plt.axis("off")
        plt.show()

    else:
        print("Invalid choice.")

def apply_edge_detection(image_array):
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]])

    sobel_y = np.array([[-1, -2, -1],
                        [0,  0,  0],
                        [1,  2,  1]])

    def convolve2d(channel, kernel):
        k_h, k_w = kernel.shape
        pad_h, pad_w = k_h // 2, k_w // 2
        padded = np.pad(channel, ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')
        result = np.zeros_like(channel, dtype=np.float64)
        for i in range(channel.shape[0]):
            for j in range(channel.shape[1]):
                result[i, j] = np.sum(padded[i:i+k_h, j:j+k_w] * kernel)
        return result

    if len(image_array.shape) == 3:
        gray = 0.299 * image_array[..., 0] + 0.587 * image_array[..., 1] + 0.114 * image_array[..., 2]
    else:
        gray = image_array

    gx = convolve2d(gray, sobel_x)
    gy = convolve2d(gray, sobel_y)

    magnitude = np.sqrt(gx**2 + gy**2)
    magnitude = np.clip(magnitude, 0, 255).astype(np.uint8)

    plt.figure(figsize=(10, 6))
    plt.subplot(121)
    plt.imshow(image_array.astype(np.uint8))
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(122)
    plt.imshow(magnitude, cmap='gray')
    plt.title("Edge Detected")
    plt.axis("off")
    plt.show()

def main():
    actions = ["Image Filters", "Blurring", "Sharpening", "Image Effects", "Edge Detection"]
    print("What would you like to do? ")
    for index, action in enumerate(actions):
        print(f"{index + 1}. {action}")

    action_index = input("Enter the index: ")
    image_array = np.load("original_image.npy")

    if action_index == "1":
        apply_image_filter(image_array)

    elif action_index == "2":
        apply_blurring(image_array)

    elif action_index == "3":
        apply_blurring(image_array)

    elif action_index == "4":
        apply_image_effects(image_array)

    elif action_index == "5":
        apply_edge_detection(image_array)

    else:
        print("Incorrect index!")
        return

if __name__ == '__main__':
    main()
