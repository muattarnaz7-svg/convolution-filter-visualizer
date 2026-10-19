"""
Convolution Filter Visualizer
-------------------------------
CS231n Lecture 5 ("What do Conv filters learn?") ka hands-on demo.

Image upload karo, aur dekho ke different hand-crafted filters (jaise
Sobel edge detector, blur, sharpen) image par convolution operation
kaise apply karte hain -- bilkul jaise lecture mein dikhaya gaya tha
ke first-layer conv filters "local image templates" (edges, oriented
patterns) seekhte hain.

Run:
    pip install streamlit numpy pillow scipy --break-system-packages
    streamlit run convolution_filter_visualizer.py
"""

import numpy as np
import streamlit as st
from PIL import Image
from scipy.signal import convolve2d

st.set_page_config(page_title="Convolution Filter Visualizer", layout="wide")

# ----------------------------------------------------------------------
# Filter bank -- har filter ek chota kernel (matrix) hai, bilkul jaisa
# lecture mein "5x5x3 filter" ya "3x3 filter" dikhaya gaya tha.
# ----------------------------------------------------------------------
FILTERS = {
    "Identity (no change)": np.array([
        [0, 0, 0],
        [0, 1, 0],
        [0, 0, 0],
    ], dtype=np.float32),

    "Edge Detection (Sobel X)": np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1],
    ], dtype=np.float32),

    "Edge Detection (Sobel Y)": np.array([
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1],
    ], dtype=np.float32),

    "Edge Detection (Laplacian)": np.array([
        [0, 1, 0],
        [1, -4, 1],
        [0, 1, 0],
    ], dtype=np.float32),

    "Box Blur": np.ones((5, 5), dtype=np.float32) / 25.0,

    "Gaussian Blur": np.array([
        [1, 4, 6, 4, 1],
        [4, 16, 24, 16, 4],
        [6, 24, 36, 24, 6],
        [4, 16, 24, 16, 4],
        [1, 4, 6, 4, 1],
    ], dtype=np.float32) / 256.0,

    "Sharpen": np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0],
    ], dtype=np.float32),

    "Emboss": np.array([
        [-2, -1, 0],
        [-1, 1, 1],
        [0, 1, 2],
    ], dtype=np.float32),

    "Outline": np.array([
        [-1, -1, -1],
        [-1, 8, -1],
        [-1, -1, -1],
    ], dtype=np.float32),
}


def apply_filter_to_channel(channel: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Ek single color channel (grayscale 2D array) par manual
    convolution apply karta hai -- 'same' padding ke sath, bilkul
    jaisa lecture mein 'Same padding: P = (K-1)/2' dikhaya gaya tha."""
    result = convolve2d(channel, kernel, mode="same", boundary="symm")
    return result


def apply_filter(image_array: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """RGB image ke teeno channels par filter independently apply
    karta hai (jaise lecture mein 'filters always extend the full
    depth of the input volume' -- yahan hum simplicity ke liye har
    channel alag se process kar rahe hain taake result visually
    samajhna asaan ho)."""
    if image_array.ndim == 2:  # already grayscale
        out = apply_filter_to_channel(image_array, kernel)
        return np.clip(out, 0, 255).astype(np.uint8)

    channels = []
    for c in range(image_array.shape[2]):
        out_c = apply_filter_to_channel(image_array[:, :, c], kernel)
        channels.append(out_c)
    stacked = np.stack(channels, axis=2)
    return np.clip(stacked, 0, 255).astype(np.uint8)


# ----------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------
st.title("🔍 Convolution Filter Visualizer")
st.caption(
    "CS231n Lecture 5 ka concept: dekho ke convolution filters image par "
    "kaise kaam karte hain -- edges detect karna, blur karna, sharpen karna."
)

uploaded_file = st.file_uploader(
    "Image upload karo (jpg/png)", type=["jpg", "jpeg", "png"]
)

with st.sidebar:
    st.header("⚙️ Settings")
    use_grayscale = st.checkbox("Grayscale mein convert karo (edges clearer dikhte hain)", value=True)
    resize_width = st.slider("Image width (resize, speed ke liye)", 100, 600, 300, step=50)

    st.markdown("---")
    st.subheader("Custom Kernel (optional)")
    use_custom = st.checkbox("Apna kernel banao")
    custom_kernel = None
    if use_custom:
        st.write("3x3 kernel values daalein:")
        rows = []
        for i in range(3):
            cols = st.columns(3)
            row = [cols[j].number_input(f"k{i}{j}", value=0.0, step=0.5, key=f"k{i}{j}") for j in range(3)]
            rows.append(row)
        custom_kernel = np.array(rows, dtype=np.float32)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    # resize for speed + consistent display
    w_percent = resize_width / float(image.size[0])
    h_size = int(float(image.size[1]) * w_percent)
    image = image.resize((resize_width, h_size))

    if use_grayscale:
        image_for_filter = image.convert("L")
        img_array = np.array(image_for_filter, dtype=np.float32)
    else:
        img_array = np.array(image, dtype=np.float32)

    st.subheader("Original Image")
    st.image(image, width=resize_width)

    st.subheader("Filters Applied")

    filter_names = list(FILTERS.keys())
    if use_custom and custom_kernel is not None:
        filter_names = ["My Custom Kernel"] + filter_names

    # Grid layout: 3 columns
    n_cols = 3
    cols = st.columns(n_cols)

    for idx, fname in enumerate(filter_names):
        kernel = custom_kernel if fname == "My Custom Kernel" else FILTERS[fname]
        filtered = apply_filter(img_array, kernel)

        col = cols[idx % n_cols]
        with col:
            st.image(filtered, caption=fname, width=resize_width, clamp=True)
            with st.expander(f"Kernel: {fname}"):
                st.write(kernel)

    st.markdown("---")
    st.info(
        "💡 **Lecture se connection:** Ye jo hand-crafted filters hum manually "
        "define kar rahe hain (Sobel, Laplacian, etc.), CNN inhe khud training "
        "ke dauran **seekh leta hai** -- bina kisi ko manually likhe! First-layer "
        "filters aksar edges/oriented patterns detect karna seekhte hain "
        "(bilkul jaisa yahan Sobel/Laplacian kar rahe hain), aur deeper layers "
        "zyada complex structures (shapes, textures) seekhti hain."
    )
else:
    st.info("👆 Upar se koi image upload karein shuru karne ke liye.")
    st.markdown(
        """
        ### Ye app kya karti hai?
        1. Aap koi bhi image upload karti hain
        2. App har filter (Sobel, Blur, Sharpen, etc.) ko ek **3x3 ya 5x5 kernel**
           ki tarah image par **convolve** (slide + dot product) karti hai
        3. Result dikhati hai ke har filter image ko kaise different tareeke se
           "dekh" raha hai -- bilkul CNN ke **first conv layer** ki tarah
        """
    )