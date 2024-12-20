import io
import os
from fastapi import HTTPException, Query, Request
import modal
import modal.gpu
import logging
from starlette.responses import Response
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

MODEL_DIR = "/model"  # Path inside the volume
volume = modal.Volume.from_name("sd3-medium", create_if_missing=True)

model_id = "stabilityai/stable-diffusion-3-medium-diffusers"

@app.cls(image=image, gpu="A10G",
 timeout=8 * ONE_MINUTE,
 secrets=[modal.Secret.from_name("huggingface-secret"),
 modal.Secret.from_name("API_ACCESS")],
 volumes={MODEL_DIR: volume})
class Inference:

    @modal.build()
    def download_model(self):
        """Downloads the model and saves it to the Modal Volume during build."""
        model_path = os.path.join(MODEL_DIR, "model_index.json")
        if os.path.exists(model_path):
            logging.info(" Skip download --> Model is already present on volume 👍")
        else:
            pipeline = StableDiffusion3Pipeline.from_pretrained(model_id, torch_dtype=torch.float16)
            pipeline.save_pretrained(MODEL_DIR)
            logging.info(" Model downloaded and saved to the volume. 🥳")
        self.API_KEY= os.environ(["API_KEY"])
    
    @modal.enter()
    def initialize(self):
        """Loads the model from the volume into GPU memory at runtime."""
        logging.info("Initializing DiffusionPipeline and loading model to GPU 🚀...")
        # Mount the volume and load model
        self.pipe = StableDiffusion3Pipeline.from_pretrained(MODEL_DIR, torch_dtype=torch.float16)
        self.pipe.enable_model_cpu_offload()
        self.pipe.enable_attention_slicing()

        logging.info(" Model successfully loaded into GPU memory. 👍")

    @modal.method()
    def run(self, prompt: str) -> list[bytes]:
        """Generates an image based on the given prompt."""
        logging.info(f"Generating image with prompt: {prompt}")
        image = self.pipe(
            prompt,
            negative_prompt="bad hands, bad feet bad face, foggy, unclear, noisy",
            num_inference_steps=15,
            guidance_scale=7.0,
        ).images[0]
        buffer = io.BytesIO()
        image.save(buffer, format="PNG", quality=95)
        return buffer.getvalue()

    @modal.web_endpoint(docs=True)
    def web(self, request: Request, prompt: str = Query(..., description="prompt for image generation")):
        """Exposes a web endpoint for generating images."""
        
        user_key = request.headers.get("X-API-KEY") 
        if user_key!= self.API_KEY:
            raise HTTPException(
                status_code=401,
                detail="Unauthorized attempt to access the endpoint"
            )

        image_bytes = self.run.local(prompt)
        return Response(
            content=image_bytes,
            status_code=200,
            media_type="image/png"
        )
