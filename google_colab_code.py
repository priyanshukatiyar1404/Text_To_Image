# !pip install streamlit diffusers ipython accelerate transformers pyngrok

from diffusers import DiffusionPipeline
import torch
from IPython.display import display



pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0",
                                         torch_dtype=torch.float16,
                                         use_safetensors=True,
                                         variant="fp16")
pipe.to("cuda")

prompt="A car on Mars"


images = pipe(prompt=prompt).images

for image in images:
  display(image)