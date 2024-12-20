import io
import os
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

    @modal.method()
    def run_with_progress(self, prompt: str):
        """Streams intermediate images during generation."""
        images_queue = queue.Queue()

        def callback_on_step_end(pipeline, step: int, timestep: int, callback_kwargs: Dict):
            latents = callback_kwargs.get("latents")
            if latents is not None:
                with torch.no_grad():
                    # Decode latents to image
                    decoded = pipeline.vae.decode(latents.to(pipeline.vae.dtype))
                    image_tensor = decoded.sample  # This is a tensor

                    # Convert to CPU numpy array for `numpy_to_pil`
                    image_tensor = image_tensor.cpu().permute(0, 2, 3, 1).float().numpy()
                
                pil_image = pipeline.numpy_to_pil(image_tensor)[0]

                # Convert PIL image to PNG bytes
                buffer = io.BytesIO()
                pil_image.save(buffer, format="PNG", quality=95)
                images_queue.put(buffer.getvalue())
            
            # Always return a dict containing the latents so the pipeline won't fail
            return {"latents": latents}


        def run_pipeline():
            self.pipe(
                prompt=prompt,
                num_inference_steps=28,
                callback_on_step_end=callback_on_step_end,
                callback_on_step_end_tensor_inputs=["latents"],
            )

        # Run the pipeline in a separate thread so we can yield images in real-time
        pipeline_thread = threading.Thread(target=run_pipeline)
        pipeline_thread.start()

        # Stream images
        boundary = b"--frame"
        while pipeline_thread.is_alive() or not images_queue.empty():
            try:
                image_data = images_queue.get(timeout=1)
                # Yield bytes directly
                yield boundary + b"\r\n" \
                      b"Content-Type: image/png\r\n\r\n" + image_data + b"\r\n"
            except queue.Empty:
                pass

    @modal.web_endpoint(docs=True)
    def web(self, prompt: str):
        """Streams image generation progress to the web."""
        return StreamingResponse(
            self.run_with_progress.local(prompt),
            media_type="multipart/x-mixed-replace;boundary=frame"
        )
