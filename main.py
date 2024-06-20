import streamlit as st
from model import load_model

st.title("Stable Diffusion XL Image Generation")
prompt = st.text_input("Enter your prompt:")

if st.button("Generate Image"):
    with st.spinner("Generating image..."):
        pipe = load_model()
        images = pipe(prompt=prompt).images
        for image in images:
            st.image(image, use_column_width=True)
        