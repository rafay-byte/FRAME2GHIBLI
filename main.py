import cv2
import numpy as np
import torch
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
import os
from tqdm import tqdm

def get_adaptive_canny_thresholds(image):
    """Calculates adaptive Canny thresholds based on image median."""
    gray_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    median = np.median(gray_image)
    # Set thresholds based on the median, ensuring they are not too tight or too wide
    low_threshold = int(max(0, 0.7 * median))
    high_threshold = int(min(255, 1.4 * median))
    return low_threshold, high_threshold

def create_canny_edge(image):
    """Create Canny edge map with adaptive thresholds for better consistency."""
    low, high = get_adaptive_canny_thresholds(image)
    canny_image = cv2.Canny(np.array(image), low, high)
    canny_image = Image.fromarray(canny_image).convert("RGB")
    return canny_image

def load_pipeline():
    """Loads the diffusion model and ControlNet."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-canny",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    )

    # IMPORTANT: Ensure you have an anime-style model.
    # 'anything-v5' is great. If it's not found, 'stable-diffusion-v1-5' will produce less anime-like results.
    try:
        # Make sure this path to your model is correct
        pipe = StableDiffusionControlNetPipeline.from_single_file(
            "./models/base/anything-v5.safetensors",
            controlnet=controlnet,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            use_safetensors=True,
            safety_checker=None
        )
        print("Successfully loaded local anime model.")
    except Exception as e:
        print(f"Could not load local model: {e}")
        print("Falling back to runwayml/stable-diffusion-v1-5. Results may not be the desired anime style.")
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            controlnet=controlnet,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            safety_checker=None
        )

    pipe.to(device)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    if device == "cuda":
        pipe.enable_model_cpu_offload()

    return pipe

def generate_frames():
    """Processes input frames to generate anime-styled output frames."""
    pipe = load_pipeline()
    input_dir = "input_frames"
    output_dir = "output_frames"
    os.makedirs(output_dir, exist_ok=True)

    frame_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    if not frame_files:
        print(f"No image files found in '{input_dir}' directory!")
        return

    # --- REFINED PROMPTS ---
    # Focus on the *style* and *aesthetic*, not the subject matter.
    positive_prompt = (
        "masterpiece, best quality, vibrant anime style, cel shading, clean bold lines, "
        "saturated colors, anime screenshot, studio ghibli style, flat colors, 2d animation"
    )

    negative_prompt = (
        "photorealistic, photography, 3d, realism, realistic, blurry, grainy, noisy, "
        "watermark, text, signature, ugly, distorted"
    )

    # --- REFINED PARAMETERS ---
    generator = torch.Generator(device=pipe.device).manual_seed(42) # Fixed seed for consistency
    guidance_scale = 7.5      # Optimal balance between prompt and image
    controlnet_scale = 0.9    # Strong guidance from Canny edges
    inference_steps = 25      # Good balance of speed and quality

    for frame_file in tqdm(frame_files, desc="🎨 Converting to Anime Style"):
        image_path = os.path.join(input_dir, frame_file)
        original_image = Image.open(image_path).convert("RGB").resize((512, 512))

        # Use adaptive Canny edge detection for each frame
        canny_image = create_canny_edge(original_image)

        result = pipe(
            prompt=positive_prompt,
            negative_prompt=negative_prompt,
            image=canny_image,
            num_inference_steps=inference_steps,
            generator=generator,
            guidance_scale=guidance_scale,
            controlnet_conditioning_scale=controlnet_scale,
        ).images[0]

        output_path = os.path.join(output_dir, frame_file)
        result.save(output_path)

    print(f"✅ Conversion complete! Check the '{output_dir}' folder.")

# --- Main execution ---
if __name__ == "__main__":
    # Ensure your source frames are in a folder named 'input_frames'
    generate_frames()