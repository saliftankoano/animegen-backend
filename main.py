import io
import os
import time
from typing import Dict
from fastapi.responses import StreamingResponse
import modal
import modal.gpu
import logging
from starlette.responses import Response
import threading
import queue

ONE_MINUTE = 60

# Initialize the modal app
app = modal.App(name="Diffusion_1")

# Setup logging configuration
logging.basicConfig(level=logging.INFO)

# Modal image with dependencies
image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "peft",
    "accelerate==0.33.0",
    "diffusers==0.31.0",
    "fastapi[standard]==0.115.4",
    "huggingface-hub[hf_transfer]==0.25.2",
    "sentencepiece==0.2.0",
    "torch==2.5.1",
    "transformers~=4.44.0",
    "bitsandbytes",
    "slowapi",
).env({
    "HF_HUB_ENABLE_HF_TRANSFER": "1",  # Faster downloads of Hugging Face models
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"  # Avoid memory segmentation
})

with image.imports():
    import torch
    from diffusers import StableDiffusion3Pipeline

MODEL_DIR = "/model"
volume = modal.Volume.from_name("sd3-medium", create_if_missing=True)
model_id = "stabilityai/stable-diffusion-3-medium-diffusers"

@app.cls(image=image, gpu="A10G", timeout=8 * ONE_MINUTE, secrets=[modal.Secret.from_name("huggingface-secret")], volumes={MODEL_DIR: volume})
class Inference:

    @modal.build()
    def download_model(self):
        """Downloads the model and saves it to the Modal Volume during build."""
        logging.info("Downloading the Stable Diffusion model 🚶‍♂️...")
        model_path = os.path.join(MODEL_DIR, "model_index.json")
        if os.path.exists(model_path):
            logging.info("Model is already present on volume 👍")
        else:
            pipeline = StableDiffusion3Pipeline.from_pretrained(model_id, torch_dtype=torch.float16)
            pipeline.save_pretrained(MODEL_DIR)
            logging.info("Model downloaded and saved to the volume. 🥳")

    @modal.enter()
    def initialize(self):
        """Loads the model from the volume into GPU memory at runtime."""
        logging.info("Initializing DiffusionPipeline and loading model to GPU 🚀...")
        self.pipe = StableDiffusion3Pipeline.from_pretrained(MODEL_DIR, torch_dtype=torch.float16)
        self.pipe.enable_model_cpu_offload()
        self.pipe.enable_attention_slicing()
        logging.info("Model successfully loaded into GPU memory. 👍")

    def generate_image(self, prompt: str, steps: int, seed: int = 42) -> bytes:
        # Create a torch.Generator and set a manual seed
        generator = torch.Generator(device=self.pipe.device)
        generator.manual_seed(seed)

        # Run the pipeline with a fixed seed
        image = self.pipe(
            prompt=prompt,
            num_inference_steps=steps,
            guidance_scale=7.0,
            generator=generator
        ).images[0]

        # Convert PIL to bytes
        buffer = io.BytesIO()
        image.save(buffer, format="PNG", quality=95)
        return buffer.getvalue()
    
    @modal.method()
    def generate_staged_images(self, prompt: str):
        """
        Pre-generate images at multiple steps to simulate progressive refinement.
        For example:
          - First run at 10 steps (rough draft)
          - Second run at 20 steps (more refined)
          - Final run at 28 steps (final image)
        """
        steps = [10, 20, 28]
        images = []
        for step in steps:
            logging.info(f"Generating image at {steps} steps...")
            image_bytes = self.generate_image(prompt=prompt, steps=step)
            images.append(image_bytes)
        return images

    @modal.web_endpoint(docs=True)
    def web(self, prompt: str):
        """Pretend to stream by sending pre-generated images one after another with a delay."""
        images = self.generate_staged_images.local(prompt)
        boundary = b"frame"

        def image_stream():
            for img in images:
                yield b"--" + boundary + b"\r\n"
                yield b"Content-Type: image/png\r\n\r\n" + img + b"\r\n"
                # Wait a bit before sending the next image
                time.sleep(1)
            # Optionally, close the multipart
            yield b"--" + boundary + b"--\r\n"

        return StreamingResponse(
            image_stream(),
            media_type="multipart/x-mixed-replace;boundary=frame"
        )
