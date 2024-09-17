import streamlit as st
import requests
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from sklearn.decomposition import TruncatedSVD


def svd_transform(image, k):
    U, sing_values, V = np.linalg.svd(image)
    sigma = np.zeros(shape=image.shape)
    np.fill_diagonal(sigma, sing_values)
    trunc_U = U[:, :k]
    trunc_sigma = sigma[:k, :k]
    trunc_V = V[:k, :]
    new_image = trunc_U@trunc_sigma@trunc_V
    return new_image

url = st.text_input('Введите URL-адрес изображения:')
if url:
    response = requests.get(url)
    image = Image.open(BytesIO(response.content))
    image = np.array(image.convert('L'))

    st.image(image, caption='Исходное изображение', use_column_width=True)

    k = st.slider('Выберите количество сингулярных чисел:', 1, min(image.shape[0], image.shape[1]), 100)

    new_image = svd_transform(image, k)

    new_image = Image.fromarray(np.uint8(new_image))

    st.image(new_image, caption=f'Изображение с {k} сингулярными числами', use_column_width=True)

