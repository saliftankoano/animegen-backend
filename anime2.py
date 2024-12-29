import modal
import modal.gpu
import logging
ONE_MINUTE = 60

# Initialize the modal app
app = modal.App(name="animegen")

# Setup logging configuration
logging.basicConfig(level=logging.INFO)

# Modal image with dependencies
image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "peft",
    "accelerate==0.33.0",
    "diffusers==0.31.0",
    "fastapi==0.115.4",
    "huggingface-hub[hf_transfer]==0.25.2",
    "sentencepiece==0.2.0",
    "torch==2.5.1",
    "transformers~=4.44.0",
    "bitsandbytes",
    "slowapi",
    "starlette",
    "requests",
).env({
    "HF_HUB_ENABLE_HF_TRANSFER": "1",  # Faster downloads of Hugging Face models
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"  # Avoid memory segmentation
})

import os
import io
from starlette.requests import Request
MODEL_DIR = "/model"  # Path inside the volume
volume = modal.Volume.from_name("animagine-xl-3.1", create_if_missing=True)

model_id = "cagliostrolab/animagine-xl-3.1"

@app.cls(image=image, gpu="A10G",timeout=8 * ONE_MINUTE, secrets=[modal.Secret.from_name("huggingface-secret"), modal.Secret.from_name("API_KEY")], volumes={MODEL_DIR: volume}, container_idle_timeout= 180)
class AnimeTest:
    @modal.build()
    def download_model(self):
        from diffusers import DiffusionPipeline
        import torch
        """Downloads the model and saves it to the Modal Volume during build."""
        model_path = os.path.join(MODEL_DIR, "model_index.json")
        if os.path.exists(model_path):
            logging.info(" Skip download --> Model is already present on volume 👍")
        else:
            pipeline = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
            pipeline.save_pretrained(MODEL_DIR)
            logging.info(" Model downloaded and saved to the volume. 🥳")

    @modal.enter()
    def initialize(self):
        """Loads the model from the volume into GPU memory at runtime."""
        import torch
        from diffusers import DiffusionPipeline
        self.API_KEY = os.environ["API_KEY"]
        logging.info("Initializing DiffusionPipeline and loading model to GPU 🚀...")
        # Mount the volume and load model
        self.pipe = DiffusionPipeline.from_pretrained(
            MODEL_DIR, 
            torch_dtype=torch.float16,
            truncation=False,
            max_length=512
        )
        self.pipe.enable_model_cpu_offload()
        self.pipe.enable_attention_slicing()

        if hasattr(self.pipe.tokenizer, 'model_max_length'):
            logging.info(f"Tokenizer max length: {self.pipe.tokenizer.model_max_length}")

        logging.info(" Model successfully loaded into GPU memory. 👍")

    @modal.method()
    def run(self, prompt: str) -> list[bytes]:
        """Generates an image based on the given prompt."""
        logging.info(f"Generating image with prompt: {prompt}")
        image = self.pipe(
            prompt,
            negative_prompt="nsfw, longbody, lowres, bad anatomy, bad hands, missing fingers, pubic hair, extra digit, fewer digits, cropped, worst quality, low quality, very displeasing",
            num_inference_steps=50,
            # guidance_scale=7.0,
        ).images[0]
        buffer = io.BytesIO()
        image.save(buffer, format="PNG", quality=95)
        return buffer.getvalue()

    @modal.web_endpoint(docs=True)
    def generate(self, prompt: str, request: Request):
        from starlette.responses import Response
        
        # api_key = request.headers.get("X-API-KEY")
        # # Validate the API key
        # if api_key != self.API_KEY:
        #     return Response("Unauthorized attempt to access the endpoint", status_code=401)
        # Generate the image
        image_bytes = self.run.local(prompt)
        return Response(content=image_bytes, status_code=200, media_type="image/png")


    @modal.web_endpoint(docs=True)
    def health(self):
        from datetime import datetime, timezone
        "Keeps the container warm"
        return {"status": "Healthy", "timestamp": datetime.now(timezone.utc).isoformat()}

@app.function(schedule=modal.Cron("0 */4 * * *"), secrets=[modal.Secret.from_name("API_KEY"), modal.Secret.from_name("HEALTH")], image=image)  # run every 4 hours
def update_keep_warm():
    import requests
    
    health_url = os.environ["HEALTH"]
    health_response = requests.get(health_url)
    print(f"Health check at: {health_response.json()['timestamp']}")
