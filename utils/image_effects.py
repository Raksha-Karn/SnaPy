import numpy as np
import matplotlib.pyplot as plt

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
