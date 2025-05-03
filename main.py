import numpy as np
from .utils.image_filters import apply_image_filter
from .utils.image_effects import apply_image_effects
from .utils.edge_detection import apply_edge_detection
from .utils.blurring_sharpening import apply_blurring, apply_sharpening

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
        apply_sharpening(image_array)

    elif action_index == "4":
        apply_image_effects(image_array)

    elif action_index == "5":
        apply_edge_detection(image_array)

    else:
        print("Incorrect index!")
        return

if __name__ == '__main__':
    main()
