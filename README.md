
# 🖼️ SnaPy - Image Processor CLI

A minimal yet powerful command-line image processing tool using **NumPy** and **Matplotlib**. Perform operations like filters, effects, blurring, sharpening, and edge detection directly on `.npy` image arrays.
## ✨ Features

- 🎨 **Image Filters** — Enhance your images with filters like Sepia, Grayscale, Negative, Posterize and Solarize.
- 🌫️ **Blurring** — Apply smoothing effects with Gaussian blur.
- 🔪 **Sharpening** — Accentuate details and edges.
- 💫 **Image Effects** — Transform images with effects like Oil Painting and Cartoon Effects.
- 🧠 **Edge Detection** — Highlight the structure and boundaries within images.

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/image-processor-cli.git
```
### 2. Install Dependencies
```bash
pip install -r requirements.txt
```
### 3. Add your image in the root directory and modify image name in the img_to_array.py file
```bash
original_image = Image.open('your_image_path').convert("RGBA")
```
### 4. Run the code
```bash
python main.py
```
