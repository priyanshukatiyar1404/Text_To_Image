# Library imports

# Importing PyTorch library, for building and training neural networks
import torch

# Importing StableDiffusionPipeline to use pre-trained Stable Diffusion models
from diffusers import StableDiffusionPipeline

# Image is a class for the PIL module to visualize images in a Python Notebook
from PIL import Image

import time

# Creating pipeline 
# use torch.float32 for CPU, half floats (float16) not supported on Intel for this module
pipeline = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4",
                                                  torch_dtype=torch.float32)

# only do 1 image for CPU let's not get too crazy

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

n_images = 1 # Let's generate 6 images based on the prompt below
prompt = ['Motorscooter Vespa on street in Los Angeles with palm trees'] * n_images

images = pipeline(prompt).images

grid = image_grid(images, rows=1, cols=1)

# save to file since if we run as cmd line we dont get an image popup
# may be a config thing
grid.save("outgen" + str(time.time()) + ".jpg")