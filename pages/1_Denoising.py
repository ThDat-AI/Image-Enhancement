import streamlit as st
import cv2
import numpy as np
import os

from utils.add_noise import add_sp_noise, add_gausian_noise
from utils.denoise_filters import denoise_mean, denoise_median, denoise_gaussian

# --- Common functions (can be moved to a shared_utils.py if many pages use them) ---
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
            st.image(img, caption=title, use_container_width=True)

# --- Helper to ensure session_state keys exist ---
def ensure_session_state_keys():
    if "original_img" not in st.session_state:
        st.session_state["original_img"] = None
    if "noisy_img" not in st.session_state:
        st.session_state["noisy_img"] = None
    if "processed_img" not in st.session_state:
        st.session_state["processed_img"] = None
    if "noise_type" not in st.session_state:
        st.session_state["noise_type"] = "None"
    if "noise_params" not in st.session_state:
        st.session_state["noise_params"] = {}
    if "denoise_params" not in st.session_state:
        st.session_state["denoise_params"] = {}

# --- Main function for the Denoising page ---
def app():
    ensure_session_state_keys()

    st.title("Denoising 🧼")
    st.write("Remove noise from your images using various techniques.")

    # --- Image Upload Section ---
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

    # Save original to session_state
    st.session_state["original_img"] = original_img.copy()

    st.markdown("---") # Separator

    # --- Step 1: Add Noise ---
    st.subheader("2. Add Noise to the Image")
    noise_type = st.selectbox(
        "Select Noise Type to Add:",
        ["None", "Salt-and-Pepper", "Gaussian"],
        key="select_noise_type"
    )
    
    noisy_img = original_img.copy()
    noise_description = "None"
    # reset stored noise params
    st.session_state["noise_params"] = {}

    if noise_type == "Salt-and-Pepper":
        st.markdown("**Salt-and-Pepper Noise Parameters**")
        col_salt, col_pepper = st.columns(2)
        with col_salt:
            salt_rate = st.slider("Salt Rate", 0.0, 0.2, 0.04, 0.01, key="salt_rate_denoise_main")
        with col_pepper:
            pepper_rate = st.slider("Pepper Rate", 0.0, 0.2, 0.06, 0.01, key="pepper_rate_denoise_main")
        noisy_img = add_sp_noise(original_img, salt_rate, pepper_rate)
        noise_description = f"Salt-and-Pepper (Salt: {salt_rate}, Pepper: {pepper_rate})"
        st.session_state["noise_params"] = {"salt_rate": salt_rate, "pepper_rate": pepper_rate}
    elif noise_type == "Gaussian":
        st.markdown("**Gaussian Noise Parameters**")
        col_mean, col_std = st.columns(2)
        with col_mean:
            mean = st.slider("Mean", -50, 50, 0, key="mean_gaussian_denoise_main")
        with col_std:
            std = st.slider("Standard Deviation", 0, 100, 15, key="std_gaussian_denoise_main")
        noisy_img = add_gausian_noise(original_img, mean, std)
        noise_description = f"Gaussian (Mean: {mean}, Std: {std})"
        st.session_state["noise_params"] = {"mean": mean, "std": std}
    
    st.session_state["noise_type"] = noise_type
    st.session_state["noisy_img"] = noisy_img.copy()

    st.caption("Original Image vs. Image with Noise")
    display_images_in_row([original_img, noisy_img], ["Original Image", f"Image with {noise_description} Noise"])

    st.markdown("---") # Separator

    # --- Step 2: Denoising Application ---
    st.subheader("3. Apply Denoising Filter")
    denoise_method = st.selectbox(
        "Select Denoising Method:",
        ["None", "Mean Filter", "Median Filter", "Gaussian Blur"],
        key="select_denoise_method"
    )

    processed_img = noisy_img.copy()
    denoise_method_name = "No Denoising Applied"
    st.session_state["denoise_params"] = {}

    if denoise_method == "Mean Filter":
        ksize_mean = st.slider("Kernel Size (Mean Filter)", 3, 21, 3, 2, help="Must be an odd number.", key="ksize_mean_denoise_main_2")
        if ksize_mean % 2 == 0: ksize_mean += 1 # Ensure odd kernel size
        processed_img = denoise_mean(noisy_img, ksize_mean)
        denoise_method_name = f"Mean Filter (k={ksize_mean})"
        st.session_state["denoise_params"] = {"method": "mean", "k": ksize_mean}
    elif denoise_method == "Median Filter":
        ksize_median = st.slider("Kernel Size (Median Filter)", 3, 21, 3, 2, help="Must be an odd number.", key="ksize_median_denoise_main_2")
        if ksize_median % 2 == 0: ksize_median += 1 # Ensure odd kernel size
        processed_img = denoise_median(noisy_img, ksize_median)
        denoise_method_name = f"Median Filter (k={ksize_median})"
        st.session_state["denoise_params"] = {"method": "median", "k": ksize_median}
    elif denoise_method == "Gaussian Blur":
        col_ksize_gauss, col_sigmaX, col_sigmaY = st.columns(3)
        with col_ksize_gauss:
            ksize_gaussian = st.slider("Kernel Size (Gaussian Blur)", 3, 21, 5, 2, help="Must be an odd number for optimal results.", key="ksize_gaussian_denoise_main_2")
            if ksize_gaussian % 2 == 0: ksize_gaussian += 1 # Ensure odd kernel size
        with col_sigmaX:
            sigmaX = st.slider("SigmaX (Gaussian Blur)", 0, 10, 0, key="sigmaX_gaussian_denoise_main_2")
        with col_sigmaY:
            sigmaY = st.slider("SigmaY (Gaussian Blur)", 0, 10, 0, key="sigmaY_gaussian_denoise_main_2")
        # denoise_gaussian may accept (img, ksize, sigmaX, sigmaY) or (img, ksize, sigmaX) depending on your impl.
        # We'll attempt to call with both sigmaX and sigmaY if available.
        try:
            processed_img = denoise_gaussian(noisy_img, ksize_gaussian, sigmaX, sigmaY)
        except TypeError:
            processed_img = denoise_gaussian(noisy_img, ksize_gaussian, sigmaX)
        denoise_method_name = f"Gaussian Blur (k={ksize_gaussian}, σX={sigmaX}, σY={sigmaY})"
        st.session_state["denoise_params"] = {"method": "gaussian", "k": ksize_gaussian, "sigmaX": sigmaX, "sigmaY": sigmaY}
    
    st.session_state["processed_img"] = processed_img.copy()

    st.caption("Noisy Image vs. Denoised Image")
    display_images_in_row([noisy_img, processed_img], [f"Image with {noise_description} Noise", f"Denoised Image ({denoise_method_name})"])

    st.markdown("---") # Separator

    st.markdown("---") # Separator
    st.subheader("4. Overall comparison")

    # Lấy ảnh gốc từ session_state
    original_img_comp = st.session_state.get("original_img")
    
    # Kiểm tra xem ảnh gốc đã được tải chưa
    if original_img_comp is None:
        st.warning("Vui lòng tải ảnh lên ở Mục 1 để xem so sánh.")
        return # Dừng nếu chưa có ảnh

    # 1. Cho người dùng chọn loại nhiễu để so sánh
    comparison_noise_type = st.selectbox(
        "Select expected noise:",
        ["Salt-and-Pepper", "Gaussian"],
        key="comparison_noise_selector"
    )

    noisy_img_comp = None
    noise_title = "Noisy"

    # 2. Tạo ảnh nhiễu dựa trên lựa chọn VÀ tham số từ Mục 2
    if comparison_noise_type == "Salt-and-Pepper":
        # Lấy tham số S&P từ slider ở Mục 2
        salt = st.session_state.get("salt_rate_denoise_main", 0.04)
        pepper = st.session_state.get("pepper_rate_denoise_main", 0.06)
        noisy_img_comp = add_sp_noise(original_img_comp, salt, pepper)
        noise_title = f"S&P (s={salt}, p={pepper})"

    elif comparison_noise_type == "Gaussian":
        # Lấy tham số Gaussian từ slider ở Mục 2
        mean = st.session_state.get("mean_gaussian_denoise_main", 0)
        std = st.session_state.get("std_gaussian_denoise_main", 15)
        noisy_img_comp = add_gausian_noise(original_img_comp, mean, std)
        noise_title = f"Gaussian (m={mean}, s={std})"

    if noisy_img_comp is None:
        noisy_img_comp = original_img_comp.copy() # Dự phòng

    # 3. Tạo 3 ảnh khử nhiễu (Mean, Median, Gauss) dựa trên tham số từ Mục 3
    
    # Lấy tham số Mean Filter từ Mục 3
    k_mean = st.session_state.get("ksize_mean_denoise_main_2", 3)
    if k_mean % 2 == 0: k_mean += 1
    denoised_mean = denoise_mean(noisy_img_comp, k_mean)
    mean_title = f"Mean (k={k_mean})"

    # Lấy tham số Median Filter từ Mục 3
    k_median = st.session_state.get("ksize_median_denoise_main_2", 3)
    if k_median % 2 == 0: k_median += 1
    denoised_median = denoise_median(noisy_img_comp, k_median)
    median_title = f"Median (k={k_median})"

    # Lấy tham số Gaussian Filter từ Mục 3
    k_gauss = st.session_state.get("ksize_gaussian_denoise_main_2", 5)
    if k_gauss % 2 == 0: k_gauss += 1
    sX = st.session_state.get("sigmaX_gaussian_denoise_main_2", 0)
    sY = st.session_state.get("sigmaY_gaussian_denoise_main_2", 0)
    
    try:
        denoised_gauss = denoise_gaussian(noisy_img_comp, k_gauss, sX, sY)
    except TypeError:
        denoised_gauss = denoise_gaussian(noisy_img_comp, k_gauss, sX)
    gauss_title = f"Gauss σX={sX}, σY={sY}    k={k_gauss}"


    # 4. Hiển thị 5 ảnh trên một hàng
    st.caption("Comparison")
    images_to_show = [
        original_img_comp, 
        noisy_img_comp, 
        denoised_mean, 
        denoised_median, 
        denoised_gauss
    ]
    titles_to_show = [
        "Original", 
        noise_title, 
        mean_title, 
        median_title, 
        gauss_title
    ]
    
    display_images_in_row(images_to_show, titles_to_show)

    st.markdown("---") # Thêm một dấu ngắt cuối cùng cho đẹp

if __name__ == "__main__":
    app()
