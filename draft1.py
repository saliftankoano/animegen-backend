import io
import modal
import os

app = modal.App(name="Memes draft")


# Install the required dependencies for the image generation
image = modal.Image.debian_slim().pip_install("peft","sentencepiece", "transformers", "accelerate", "torch", "diffusers", "fastapi[standard]", "pyyaml>=5.1")

# On the image instance we then import the required libraries
with image.imports():
    from fastapi import Response
    from diffusers import DiffusionPipeline


@app.cls(image=image, gpu="T4", secrets=[modal.Secret.from_name("huggingface-secret")],)
class Generator:
    
    def initialize_model(self):
        self.pipe = DiffusionPipeline.from_pretrained("black-forest-labs/FLUX.1-dev")
        self.pipe.load_lora_weights("prithivMLmods/Flux-Meme-Xd-LoRA")
        
        return self.pipe
    
    @modal.build()
    def build(self):
        self.initialize_model()
        
    @modal.enter()
    def enter(self):
        pipe = self.initialize_model()
        pipe.to("cuda")
        return pipe
    
    @modal.method()
    def generate(self, prompt: str = "meme cat dancing on a computer"):
        image = self.pipe(prompt).images[0]
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")
        buffer.seek(0)
        return Response(content=buffer.getvalue(), media_type="image/jpeg")

