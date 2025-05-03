import numpy as np
import matplotlib.pyplot as plt

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
