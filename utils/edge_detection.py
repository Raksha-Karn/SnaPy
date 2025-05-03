import numpy as np
import matplotlib.pyplot as plt

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
