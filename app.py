import streamlit as st
import pickle
import numpy as np

from skimage import io
from skimage.transform import resize
from skimage.feature import hog
from skimage.color import rgb2gray

with open('model_svc.pkl', 'rb') as file:
    model = pickle.load(file)

st.title('Prediksi Gambar Tulisan Tangan')
uploaded_file = st.file_uploader("Pilih gambar", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    height_img = 40
    width_img = 40

    st.subheader('Proses Preprocessing Gambar')
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        image = io.imread(uploaded_file)
        resize_image = resize(image, (height_img, width_img), anti_aliasing=True)
        st.image(resize_image, caption='Gambar Resize 40 x 40', use_container_width=True)

    with col2:
        gray_image = rgb2gray(resize_image)
        st.image(gray_image, caption='Gambar Grayscale', use_container_width=True)

        gray_image_norm = rgb2gray(resize_image) / 255
        gray_image_norm = gray_image.squeeze()
    
    with col3:
        hist, vis_hog = hog(gray_image_norm, pixels_per_cell=(8, 8), cells_per_block=(3, 3), block_norm="L2-Hys", visualize=True)
        st.image(vis_hog, caption='Gambar Visualisasi HOG', use_container_width=True)

    prediction = model.predict([hist])
    
    st.write(f'Hasil Prediksi: {prediction[0]}')
