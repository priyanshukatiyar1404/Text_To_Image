import streamlit as st
from diffusers import DiffusionPipeline
import torch

@st.cache_resource
def load_model():
    pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0",
                                             torch_dtype=torch.float16,
                                             use_safetensors=True,
                                             variant="fp16")
    pipe.to("cuda")
    return pipe

pipe = load_model()

def generate_image(prompt):
    image = pipe(prompt)["sample"][0]
    return image

st.title("Stable Diffusion XL Image Generation")
prompt = st.text_input("Enter your prompt:")

if st.button("Generate Image"):
    with st.spinner("Generating image..."):
        image = generate_image(prompt)
        st.image(image)
