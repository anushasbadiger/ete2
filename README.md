# Image Color Palette Generator

A Streamlit application that extracts color palettes from uploaded images. The app allows users to adjust various image settings and displays the extracted colors in both table and visual format.

## Features

- Upload images (JPG, PNG, JPEG)
- Extract color palettes using K-means clustering
- Adjust image settings:
  - Resize
  - Rotate
  - Brightness
  - Contrast
  - Sharpness
- Choose color format (BGR or HEX)
- Reset settings functionality
- Responsive UI with sidebar controls

## Installation

1. Clone the repository:
```bash
git clone https://github.com/anushasbadiger/ete2.git
cd ete2
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

Run the Streamlit app:
```bash
streamlit run ete_2.py
```

The app will open in your default web browser at `http://localhost:8501`

## Requirements

- Python 3.7+
- OpenCV
- Pandas
- NumPy
- Streamlit
- Matplotlib
- scikit-learn
- Pillow

## License

MIT License 