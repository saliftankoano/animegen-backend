import modal
import modal.gpu
import logging
ONE_MINUTE = 60

# Initialize the modal app
app = modal.App(name="genwalls")

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

MODEL_DIR = "/model"  # Path inside the volume
volume = modal.Volume.from_name("sd3-medium", create_if_missing=True)

model_id = "stabilityai/stable-diffusion-3-medium-diffusers"

@app.cls(image=image, gpu="A10G",timeout=8 * ONE_MINUTE, secrets=[modal.Secret.from_name("huggingface-secret"), modal.Secret.from_name("API_ACCESS")], volumes={MODEL_DIR: volume}, container_idle_timeout= 300)
class Inference:
    @modal.build()
    def download_model(self):
        from diffusers import StableDiffusion3Pipeline
        import torch
        """Downloads the model and saves it to the Modal Volume during build."""
        model_path = os.path.join(MODEL_DIR, "model_index.json")
        if os.path.exists(model_path):
            logging.info(" Skip download --> Model is already present on volume 👍")
        else:
            pipeline = StableDiffusion3Pipeline.from_pretrained(model_id, torch_dtype=torch.float16)
            pipeline.save_pretrained(MODEL_DIR)
            logging.info(" Model downloaded and saved to the volume. 🥳")

    @modal.enter()
    def initialize(self):
        """Loads the model from the volume into GPU memory at runtime."""
        import torch
        from diffusers import StableDiffusion3Pipeline
        self.API_KEY = os.environ["API_KEY"]
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
            negative_prompt="malformed body, malformed face, malformed hands, malformed feet, bad hands, bad feet, bad face, foggy, unclear, noisy",
            num_inference_steps=15,
            guidance_scale=7.0,
        ).images[0]
        buffer = io.BytesIO()
        image.save(buffer, format="PNG", quality=95)
        return buffer.getvalue()

    @modal.web_endpoint(docs=True)
    def generate(self, prompt: str, api_key: str):
        from starlette.responses import Response

        # Validate the API key
        if api_key != self.API_KEY:
            return Response("Unauthorized attempt to access the endpoint", status_code=401)

        # Generate the image
        image_bytes = self.run.local(prompt)
        return Response(content=image_bytes, status_code=200, media_type="image/png")


    @modal.web_endpoint(docs=True)
    def health(self):
        from datetime import datetime, timezone
        "Keeps the container warm"
        return {"status": "Healthy", "timestamp": datetime.now(timezone.utc).isoformat()}

@app.function(schedule=modal.Cron("*/5 * * * *"), secrets=[modal.Secret.from_name("API_ACCESS")], image=image)  # run every 5 minutes
def update_keep_warm():
    from datetime import datetime, timezone
    import requests
    health_url = "https://saliftankoano--genwalls-inference-health.modal.run"
    generate_url = "https://saliftankoano--genwalls-inference-generate.modal.run"
    
    health_response = requests.get(health_url)
    print(f"Health check at: {health_response.json()['timestamp']}")

    # Send a generation 
    # Consider removal to avoid charges for GPU usage on image generations every 5 mins
    headers = {"X-API-KEY": os.environ["API_KEY"]}
    generate_response = requests.get(generate_url, headers=headers)
    print(f"Generation successful at: {datetime.now(timezone.utc).isoformat()}")