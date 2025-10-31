import streamlit as st
import cv2
import numpy as np
import os

from utils.sharpen_filter import sharp_image
from utils.denoise_filters import blur_image # Import the blur_image function

# --- Common functions ---
DEFAULT_IMAGE_PATH = "assets/images/default_image.jpg"
if not os.path.exists(DEFAULT_IMAGE_PATH):
    st.error(f"Error: Default image not found at {DEFAULT_IMAGE_PATH}. Please ensure the 'assets/images' directory and 'default_image.jpg' exist.")
    st.stop()

@st.cache_data
def load_default_image():
    img = cv2.imread(DEFAULT_IMAGE_PATH)
    if img is not None:
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return None

def get_image_from_upload(uploaded_file):
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img
    return None

def display_images_in_row(images, titles):
    cols = st.columns(len(images))
    for i, (img, title) in enumerate(zip(images, titles)):
        with cols[i]:
            st.image(img, caption=title,use_container_width=True)

# --- Main function for the Sharpening page ---
def app():
    st.title("Sharpening 🔪")
    st.write("First, intentionally blur an image, then apply sharpening to see the effect.")

    # --- 1. Image Upload Section ---
    st.subheader("1. Upload Your Image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        original_img = get_image_from_upload(uploaded_file)
    else:
        original_img = load_default_image()
        st.info("Using a default image. Upload your own image above if you prefer.")

    if original_img is None:
        st.error("Could not load image. Please check the default image path or try uploading another file.")
        return

    st.markdown("---") # Separator

    # --- 2. Blur the Image (for demonstration) ---
    st.subheader("2. Blur the Image (Preparation for Sharpening)")
    st.write("Apply a blur filter to the original image to simulate a soft image.")

    blur_ksize = st.slider(
        "Blur Kernel Size (ksize x ksize)", 3, 21, 5, 2, 
        help="Choose an odd number for the blur kernel size. Higher values mean more blur.",
        key="blur_ksize_sharpen_prep"
    )
    if blur_ksize % 2 == 0: # Ensure odd ksize
        blur_ksize += 1

    blurred_img = blur_image(original_img, blur_ksize)

    st.caption("Original Image vs. Blurred Image")
    display_images_in_row(
        [original_img, blurred_img], 
        ["Original Image", f"Blurred Image (k={blur_ksize})"]
    )

    st.markdown("---") # Separator

    # --- 3. Sharpen the Blurred Image ---
    st.subheader("3. Apply Sharpening Filter")
    st.write("Now, apply the sharpening filter to the intentionally blurred image.")

    apply_sharpen = st.checkbox("Apply Sharpening", value=True, key="apply_sharpen_checkbox")

    sharpened_img = blurred_img.copy()
    if apply_sharpen:
        sharpened_img = sharp_image(blurred_img)
    
    st.caption("Blurred Image vs. Sharpened Image")
    display_images_in_row(
        [blurred_img, sharpened_img], 
        [f"Blurred Image (k={blur_ksize})", "Sharpened Image (from Blurred)"]
    )

    st.markdown("---") # Separator

    # --- 4. Overall Comparison ---
    st.subheader("4. Overall Comparison: Original, Blurred, and Sharpened")
    display_images_in_row(
        [original_img, blurred_img, sharpened_img],
        ["Original Image", f"Blurred (k={blur_ksize})", "Sharpened (from Blurred)"]
    )

if __name__ == "__main__":
    app()