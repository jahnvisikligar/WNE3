from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from rag_prompt_enhancer import RAGPromptEnhancer
from lora_stable_diffusion_realisticvision import LoraStableDiffusion

# Initialize FastAPI
app = FastAPI()

# Load Prompt Enhancer
attributes_csv = "attributes.csv"  # Path to attributes CSV
categories_csv = "categories.csv"  # Path to categories CSV
prompt_enhancer = RAGPromptEnhancer(attributes_csv, categories_csv)

# Load Stable Diffusion
model_path = "CompVis/stable-diffusion-v1-4"
lora_weights_path = "realisticVisionV60B1_v60B1VAE.safetensors"
stable_diffusion = LoraStableDiffusion(model_path, lora_weights_path)

# Input Model
class PromptInput(BaseModel):
    prompt: str

# API Endpoint
@app.post("/generate_image/")
def generate_image(input: PromptInput):
    try:
        # Step 1: Enhance Prompt
        enhanced_prompt = prompt_enhancer.enhance_prompt(input.prompt)

        # Step 2: Generate Image
        image_path = stable_diffusion.generate_image(enhanced_prompt)

        return {"enhanced_prompt": enhanced_prompt, "image_path": image_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)