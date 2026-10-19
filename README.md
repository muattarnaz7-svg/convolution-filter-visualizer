# 🔍 Convolution Filter Visualizer

An interactive **Streamlit** app that shows how convolution filters work on images. Upload any image and see how classic hand-crafted kernels (Sobel, Laplacian, Blur, Sharpen, Emboss and more) transform it, the same operation a CNN's first layer performs.

Built as a hands-on demo for **CS231n Lecture 5: "What do Conv filters learn?"**

---

## 📸 Demo

> Add a screenshot of the app here:
>
> `![App Screenshot](screenshot.png)`

---

## ✨ Features

- Upload your own image (JPG / PNG)
- 9 built-in filters shown side by side in a grid
- Create your own **custom 3x3 kernel** and see the result instantly
- Toggle grayscale mode (edges are clearer in grayscale)
- Adjustable image width for faster processing
- Expandable view of each filter's kernel matrix

## 🧪 Filters Included

| Filter | Kernel Size | What it does |
|---|---|---|
| Identity | 3x3 | Returns the image unchanged |
| Sobel X | 3x3 | Detects vertical edges |
| Sobel Y | 3x3 | Detects horizontal edges |
| Laplacian | 3x3 | Detects edges in all directions |
| Box Blur | 5x5 | Simple averaging blur |
| Gaussian Blur | 5x5 | Smooth, weighted blur |
| Sharpen | 3x3 | Enhances edges and detail |
| Emboss | 3x3 | Gives a 3D raised effect |
| Outline | 3x3 | Highlights object outlines |

## 🧠 How It Works

A convolution slides a small kernel (matrix) over the image. At each position it takes the dot product of the kernel with the pixels beneath it to produce one output pixel.

- **Same padding** keeps the output the same size as the input
- Each color channel is filtered independently (or one channel in grayscale mode)
- Results are clipped to the valid 0-255 pixel range

In a CNN, these kernels are not hand-written. The network **learns** them during training. First-layer filters typically end up detecting edges and oriented patterns (much like Sobel and Laplacian here), while deeper layers learn more complex shapes and textures.

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/muattarnaz7-svg/convolution-filter-visualizer.git
cd convolution-filter-visualizer
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the app

```bash
python -m streamlit run visualizer.py
```

The app opens in your browser at `http://localhost:8501`.

> **Windows tip:** if `streamlit run` is not recognized, use `python -m streamlit run` as shown above.

## 🛠️ Tech Stack

- Python
- Streamlit
- NumPy
- SciPy (`convolve2d`)
- Pillow

## 📦 requirements.txt

```
streamlit
numpy
pillow
scipy
```

## 📚 Reference

- [CS231n: Convolutional Neural Networks for Visual Recognition](https://cs231n.github.io/)

## 👤 Author

**GitHub:** [muattarnaz7-svg](https://github.com/muattarnaz7-svg)

---

⭐ If you found this useful, consider giving the repo a star!
