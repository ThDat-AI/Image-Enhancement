import streamlit as st
import os

# --- Configuration ---
st.set_page_config(
    page_title="Image Processing App",
    page_icon="🖼️",
    layout="wide"
)

# --- Home Page ---
# Streamlit will automatically list pages in the 'pages/' directory in the sidebar.
# This app.py file becomes the default/home page.

st.title("Welcome to Image Processing App 🖼️")
st.write("---")
try:
    with open("README.md", "r", encoding="utf-8") as f:
        readme_content = f.read()
    st.markdown(readme_content)
except FileNotFoundError:
    st.warning("README.md not found. Please create one for the home page content.")
st.write("---")
st.info("Navigate using the sidebar to explore different image processing functionalities.")

# You can add more content specific to the home page here.
st.subheader("How to Use:")
st.markdown("""
1.  **Select a function** from the sidebar (Denoising, Sharpening, Edge detection).
2.  **Upload your own image** or use the default image provided.
3.  **Adjust the parameters** using the sliders and select boxes in the sidebar.
4.  **See the results** instantly!
""")

# Note: Streamlit multi-page apps don't need the 'st.sidebar.radio'
# for navigation in the main app.py when pages are in the 'pages/' folder.
# It automatically generates the navigation.