# Description: This is a streamlit app that extracts the color palette from an uploaded image. The user can adjust the number of colors, color format, and image settings such as resize, rotate, brightness, contrast, and sharpness. The extracted colors are displayed in a table and color palette format.

import cv2
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from collections import Counter
from PIL import Image, ImageEnhance

# Initialize session state for reset functionality
if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None
if "num_colors" not in st.session_state:
    st.session_state.num_colors = 5
if "color_format" not in st.session_state:
    st.session_state.color_format = "BGR"
if "resize_width" not in st.session_state:
    st.session_state.resize_width = 500
if "resize_height" not in st.session_state:
    st.session_state.resize_height = 500
if "rotate_angle" not in st.session_state:
    st.session_state.rotate_angle = 0
if "brightness" not in st.session_state:
    st.session_state.brightness = 1.0
if "contrast" not in st.session_state:
    st.session_state.contrast = 1.0
if "sharpness" not in st.session_state:
    st.session_state.sharpness = 1.0

# Function to extract colors from an image
def extract_colors(image, num_colors=5, color_format="BGR"):
    try:
        # Reshape for clustering, limiting the number of pixels for large images
        max_pixels = 10000
        if image.shape[0] * image.shape[1] > max_pixels:
            # Resize image while maintaining aspect ratio
            scale = np.sqrt(max_pixels / (image.shape[0] * image.shape[1]))
            new_size = (int(image.shape[1] * scale), int(image.shape[0] * scale))
            image = cv2.resize(image, new_size)
        
        image = image.reshape((-1, 3))  # Reshape for clustering

        # Apply KMeans clustering
        kmeans = KMeans(n_clusters=num_colors, random_state=42, n_init=10)
        labels = kmeans.fit_predict(image)
        counts = Counter(labels)

        # Sort colors by frequency
        ordered_colors = [kmeans.cluster_centers_[i] for i in counts.keys()]

        # Convert colors to required format
        if color_format == "BGR":
            colors = [list(map(int, color)) for color in ordered_colors]
        else:  # HEX format
            colors = ['#{:02x}{:02x}{:02x}'.format(*map(int, color[::-1])) for color in ordered_colors]

        return colors
    except Exception as e:
        st.error(f"Error extracting colors: {str(e)}")
        return []

# Function to display color palette
def display_palette(colors, color_format):
    fig, ax = plt.subplots(figsize=(8, 2))
    ax.set_xticks([])
    ax.set_yticks([])

    # Create color bars
    for i, color in enumerate(colors):
        if color_format == "BGR":
            rect_color = np.array(color) / 255  # Normalize BGR values
        else:  # Convert HEX to RGB
            rect_color = np.array([int(color[i:i+2], 16) for i in (1, 3, 5)]) / 255  

        ax.add_patch(plt.Rectangle((i, 0), 1, 1, color=rect_color))  # Add color patch

    ax.set_xlim(0, len(colors))
    ax.set_ylim(0, 1)
    st.pyplot(fig)

# Streamlit App UI
st.title("🎨 Image Color Palette Generator")

# File uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
st.session_state.uploaded_file = uploaded_file

# Sidebar UI
st.sidebar.header("⚙️ Image Settings")

# Sidebar options for color extraction
num_colors = st.sidebar.slider("🎨 Number of Colors", 2, 20, 5)
color_format = st.sidebar.radio("📂 Color Format", ["BGR", "HEX"])

# Image Manipulation Options
st.sidebar.subheader("🖼️ Image Adjustments")
st.session_state.resize_width = st.sidebar.slider("🔍 Resize Width", 50, 1000, st.session_state.resize_width)
st.session_state.resize_height = st.sidebar.slider("🔍 Resize Height", 50, 1000, st.session_state.resize_height)
st.session_state.rotate_angle = st.sidebar.slider("🔄 Rotate Angle", -180, 180, st.session_state.rotate_angle)
st.session_state.brightness = st.sidebar.slider("☀️ Brightness", 0.5, 3.0, st.session_state.brightness)
st.session_state.contrast = st.sidebar.slider("🌓 Contrast", 0.5, 3.0, st.session_state.contrast)
st.session_state.sharpness = st.sidebar.slider("🔎 Sharpness", 0.5, 3.0, st.session_state.sharpness)
# resize_width = st.sidebar.slider("🔍 Resize Width", 50, 1000, 500)
# resize_height = st.sidebar.slider("🔍 Resize Height", 50, 1000, 500)
# rotate_angle = st.sidebar.slider("🔄 Rotate Angle", -180, 180, 0)
# brightness = st.sidebar.slider("☀️ Brightness", 0.5, 3.0, 1.0)
# contrast = st.sidebar.slider("🌓 Contrast", 0.5, 3.0, 1.0)
# sharpness = st.sidebar.slider("🔎 Sharpness", 0.5, 3.0, 1.0)

def reset_settings():
    st.session_state.uploaded_file = None
    st.session_state.num_colors = 5
    st.session_state.color_format = "BGR"
    st.session_state.resize_width = 500
    st.session_state.resize_height = 500
    st.session_state.rotate_angle = 0
    st.session_state.brightness = 1.0
    st.session_state.contrast = 1.0
    st.session_state.sharpness = 1.0
    st.rerun()  # Reload the app
    
#Reset button
if st.sidebar.button("🔄 Reset Settings"):
    reset_settings()


if uploaded_file is not None:
    try:
        # Convert to OpenCV format
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)  # Read image in BGR
        
        if image is None:
            st.error("Failed to load image. Please try a different image format.")
        else:
            # Convert OpenCV BGR to PIL RGB for processing
            image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            # Apply Adjustments
            try:
                image_pil = image_pil.resize((st.session_state.resize_width, st.session_state.resize_height))  # Resize
                image_pil = image_pil.rotate(st.session_state.rotate_angle)  # Rotate
                image_pil = ImageEnhance.Brightness(image_pil).enhance(st.session_state.brightness)  # Adjust Brightness
                image_pil = ImageEnhance.Contrast(image_pil).enhance(st.session_state.contrast)  # Adjust Contrast
                image_pil = ImageEnhance.Sharpness(image_pil).enhance(st.session_state.sharpness)  # Adjust Sharpness
            except Exception as e:
                st.error(f"Error applying image adjustments: {str(e)}")
                st.stop()  # Stop execution instead of using return

            # Convert back to OpenCV format
            image = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

            # Extract colors
            colors = extract_colors(image, num_colors, color_format)

            if colors:  # Only proceed if colors were successfully extracted
                # Display extracted colors in table
                st.write("### 🎨 Extracted Colors:")
                if color_format == "BGR":
                    color_df = pd.DataFrame(colors, columns=["B", "G", "R"])
                else:
                    color_df = pd.DataFrame([[c] for c in colors], columns=["HEX"])

                st.dataframe(color_df)

                # Display color palette
                st.write("### 🎨 Color Palette:")
                display_palette(colors, color_format)

                # Show the uploaded image
                st.image(image_pil, caption="🖼️ Processed Image", use_column_width=True)
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
