from diffusers import StableDiffusionPipeline
from safetensors.torch import load_file
import torch

class LoraStableDiffusion:
    def __init__(self, model_path, lora_weights_path):
        self.pipeline = StableDiffusionPipeline.from_pretrained(model_path)
        self.pipeline.to("cuda" if torch.cuda.is_available() else "cpu")
        self.lora_weights = load_file(lora_weights_path)
        self.apply_lora_weights()

    def apply_lora_weights(self):
        for name, param in self.pipeline.unet.named_parameters():
            if name in self.lora_weights:
                param.data += self.lora_weights[name].data

    def generate_image(self, prompt, negative_prompt="low quality, deformed, blurry", guidance_scale=7.5):
        result = self.pipeline(prompt, negative_prompt=negative_prompt, guidance_scale=guidance_scale)
        generated_image = result.images[0]
        output_path = "generated_image.png"
        generated_image.save(output_path)
        return output_path


# if __name__ == "__main__":
#     # Example usage
#     model_path = "CompVis/stable-diffusion-v1-4"  # HuggingFace model path
#     lora_weights_path = "realisticVisionV60B1_v60B1VAE.safetensors"  # Path to LoRA weights

#     stable_diffusion = LoraStableDiffusion(model_path, lora_weights_path)
#     enhanced_prompt = input("Enter the enhanced prompt: ")
#     image_path = stable_diffusion.generate_image(enhanced_prompt)
#     print(f"Generated Image saved at: {image_path}")