import streamlit as st
import cv2
import numpy as np
import os

from utils.edge_detection import edge_Sobel, edge_Prewitt, edge_Canny

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
            # Convert grayscale images to RGB for consistent display in st.image if necessary
            if len(img.shape) == 2 or img.shape[2] == 1:
                img_display = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            else:
                img_display = img
            st.image(img_display, caption=title, use_container_width=True)

# --- Main function for the Edge Detection page ---
def app():
    st.title("Edge Detection 📐")
    st.write("Identify and highlight the boundaries of objects in your images using various algorithms.")

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

    # --- 2. Select Edge Detection Method and Parameters ---
    st.subheader("2. Select Edge Detection Method and Parameters")
    
    edge_method = st.selectbox(
        "Select Edge Detection Method:",
        ["Sobel", "Prewitt", "Canny"],
        key="select_edge_method"
    )

    processed_img = original_img.copy() 
    edge_method_name = ""
    
    # Parameters for selected method
    if edge_method == "Sobel":
        ksize = st.slider("Kernel Size (Sobel)", 3, 7, 3, 2, 
                          help="Must be an odd number (3, 5, or 7). Higher values detect thicker edges.", 
                          key="ksize_sobel")
        if ksize % 2 == 0: ksize += 1 # Ensure odd kernel size
        if ksize > 7: ksize = 7 # Limit Sobel kernel size as per OpenCV
        processed_img = edge_Sobel(original_img, ksize)
        edge_method_name = f"Sobel (k={ksize})"
    elif edge_method == "Prewitt":
        # Prewitt doesn't have a ksize parameter in our utils function
        st.write("Prewitt filter does not require additional parameters.")
        processed_img = edge_Prewitt(original_img)
        edge_method_name = "Prewitt"
    elif edge_method == "Canny":
        col_low_thresh, col_high_thresh = st.columns(2)
        with col_low_thresh:
            low_threshold = st.slider("Low Threshold (Canny)", 0, 255, 100, 
                                      help="Minimum intensity gradient value.", 
                                      key="low_threshold_canny")
        with col_high_thresh:
            high_threshold = st.slider("High Threshold (Canny)", 0, 255, 200, 
                                       help="Maximum intensity gradient value. Should be 2-3 times the low threshold.", 
                                       key="high_threshold_canny")
        processed_img = edge_Canny(original_img, low_threshold, high_threshold)
        edge_method_name = f"Canny (Low: {low_threshold}, High: {high_threshold})"

    st.caption("Original Image vs. Detected Edges")
    display_images_in_row(
        [original_img, processed_img], 
        ["Original Image", f"{edge_method_name} Edges"]
    )

    st.markdown("---") # Separator

    st.subheader("3. Overall comparison")

    # --- 1. Lấy tham số và chạy SOBEL ---
    # Lấy ksize từ slider ở Mục 2, mặc định là 3 nếu chưa có
    k_sobel = st.session_state.get("ksize_sobel", 3)
    # Sao chép logic đảm bảo ksize là số lẻ và <= 7
    if k_sobel % 2 == 0: k_sobel += 1
    if k_sobel > 7: k_sobel = 7
    
    sobel_img = edge_Sobel(original_img, k_sobel)
    sobel_title = f"Sobel (k={k_sobel})"

    # --- 2. Chạy PREWITT ---
    # Phương thức Prewitt của bạn không có tham số
    prewitt_img = edge_Prewitt(original_img)
    prewitt_title = "Prewitt"

    # --- 3. Lấy tham số và chạy CANNY ---
    # Lấy thresholds từ sliders ở Mục 2
    t1_canny = st.session_state.get("low_threshold_canny", 100)
    t2_canny = st.session_state.get("high_threshold_canny", 200)
    
    canny_img = edge_Canny(original_img, t1_canny, t2_canny)
    # Sử dụng <br> để xuống dòng như bạn muốn
    canny_title = f"Canny (t1={t1_canny}, t2={t2_canny})"

    # --- 4. Hiển thị tất cả trên một hàng ---
    st.caption("Comparison")
    
    images_to_show = [
        original_img, 
        sobel_img, 
        prewitt_img, 
        canny_img
    ]
    
    titles_to_show = [
        "Original", 
        sobel_title, 
        prewitt_title, 
        canny_title
    ]
    
    # Sử dụng lại hàm của bạn để hiển thị
    display_images_in_row(images_to_show, titles_to_show)

    st.markdown("---") # Separator
    
if __name__ == "__main__":
    app()