# Library imports

# Importing PyTorch library, for building and training neural networks
import torch

# Importing StableDiffusionPipeline to use pre-trained Stable Diffusion models
from diffusers import StableDiffusionPipeline

# Image is a class for the PIL module to visualize images in a Python Notebook
from PIL import Image

import streamlit as st

# Creating pipeline 
# use torch.float32 for CPU, half floats (float16) not supported on Intel for this module
pipeline = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4",
                                                  torch_dtype=torch.float32)

# Defining function for the creation of a grid of images
def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols
    
    w, h = imgs[0].size
    grid = Image.new('RGB', size = (cols*w,
                                   rows * w))
    grid_w, grid_h = grid.size
    
    for i, img in enumerate(imgs):
        grid.paste(img, box = (i%cols*w, i // cols*h))
    return grid

def generate_images(prompt):
    images = pipeline(prompt).images
    grid = image_grid(images, rows=1, cols=len(images))
    return grid

# Streamlit app
def app():
    st.title("Stable Diffusion Image Generation")
    prompt = st.text_input("Enter your prompt:")
    if st.button("Generate Image"):
        with st.spinner("Generating image..."):
            image = generate_images([prompt])
            st.image(image)

if __name__ == "__main__":
    app()