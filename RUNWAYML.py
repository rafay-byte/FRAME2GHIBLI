from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from PIL import Image
import torch
import os
from tqdm import tqdm
import glob
import sys
import json
from datetime import datetime

# Resume/checkpoint system
CHECKPOINT_FILE = "progress_checkpoint.json"
MODEL_CACHE_DIR = "./model_cache"


def save_checkpoint(frame_index, total_frames, start_time):
    """Save current progress"""
    checkpoint = {
        "last_completed_frame": frame_index,
        "total_frames": total_frames,
        "start_time": start_time,
        "timestamp": datetime.now().isoformat()
    }
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(checkpoint, f, indent=2)
    print(f"💾 Checkpoint saved: Frame {frame_index + 1}/{total_frames}", flush=True)


def load_checkpoint():
    """Load previous progress"""
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r') as f:
            return json.load(f)
    return None


def get_completed_frames():
    """Get list of already processed frames"""
    if not os.path.exists("stylized"):
        return []
    return [f.replace('.png', '') for f in os.listdir("stylized") if f.endswith('.png')]


print("🚀 Starting Ghibli stylization with RESUME capability...")

# Check for previous progress
checkpoint = load_checkpoint()
completed_frames = get_completed_frames()

if checkpoint:
    print(f"📋 Found checkpoint: {checkpoint['last_completed_frame'] + 1}/{checkpoint['total_frames']} frames completed",
          flush=True)
    print(f"⏰ Last run: {checkpoint['timestamp']}", flush=True)

if completed_frames:
    print(f"✅ Found {len(completed_frames)} already processed frames", flush=True)

# Create directories
os.makedirs("stylized", exist_ok=True)
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

print("📦 Loading models (cached if available)...", flush=True)

try:
    # Load with caching - models won't re-download if already cached
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-depth",
        torch_dtype=torch.float16,
        cache_dir=MODEL_CACHE_DIR,
        resume_download=True  # Resume interrupted downloads
    )
    print("✅ ControlNet loaded from cache!", flush=True)

    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=controlnet,
        torch_dtype=torch.float16,
        cache_dir=MODEL_CACHE_DIR,
        resume_download=True
    ).to("cuda")
    print("✅ Pipeline loaded and moved to GPU!", flush=True)

except Exception as e:
    print(f"❌ Model loading failed: {e}", flush=True)
    print("🔄 Retrying with force download...", flush=True)
    # Fallback without cache
    controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-depth", torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16
    ).to("cuda")

# Optimizations for RTX 3080
print("⚡ Applying RTX 3080 optimizations...", flush=True)
pipe.enable_xformers_memory_efficient_attention()
pipe.enable_attention_slicing()
pipe.enable_cpu_offload()
print("✅ Optimizations applied!", flush=True)

# Get all frames and filter out completed ones
all_frames = sorted(glob.glob("frames/*.png"))
frames_to_process = []

for frame_path in all_frames:
    frame_name = os.path.basename(frame_path).replace('.png', '')
    if frame_name not in completed_frames:
        frames_to_process.append(frame_path)

total_frames = len(all_frames)
remaining_frames = len(frames_to_process)

print(f"📁 Total frames: {total_frames}", flush=True)
print(f"✅ Already completed: {len(completed_frames)}", flush=True)
print(f"🎬 Remaining to process: {remaining_frames}", flush=True)

if remaining_frames == 0:
    print("🎉 ALL FRAMES ALREADY COMPLETED! Nothing to do.", flush=True)
    sys.exit(0)

# Best settings for quality and RTX 3080
prompt = "Studio Ghibli animation style, soft colors, fantasy background, masterpiece, high quality"
negative_prompt = "blurry, low quality, distorted, ugly, deformed"

# Process with resume capability
print("🎨 Starting/Resuming stylization process...", flush=True)
start_time = datetime.now().isoformat()

try:
    for i, frame_path in enumerate(tqdm(frames_to_process, desc="🎬 Stylizing", unit="frame")):
        frame_name = os.path.basename(frame_path)
        current_frame_index = len(completed_frames) + i

        print(f"🖼️  Processing frame {current_frame_index + 1}/{total_frames}: {frame_name}", flush=True)

        # Load and process image
        image = Image.open(frame_path)
        print(f"   📏 Image size: {image.size}", flush=True)
        print(f"   🎨 Generating Ghibli style...", flush=True)

        # RTX 3080 optimized settings
        stylized = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image,
            num_inference_steps=35,  # Balanced for 3080
            guidance_scale=7.5,
            controlnet_conditioning_scale=1.0,
            generator=torch.Generator(device="cuda").manual_seed(42)  # Consistent results
        ).images[0]

        # Save result
        output_path = f"stylized/{frame_name}"
        stylized.save(output_path)
        print(f"   ✅ Saved: {output_path}", flush=True)

        # Save checkpoint every frame (in case of interruption)
        save_checkpoint(current_frame_index, total_frames, start_time)

        # Clear cache every 10 frames
        if (i + 1) % 10 == 0:
            print("🧹 Clearing GPU cache...", flush=True)
            torch.cuda.empty_cache()

except KeyboardInterrupt:
    print("\n⚠️  INTERRUPTED BY USER", flush=True)
    print(f"💾 Progress saved! Run again to resume from frame {current_frame_index + 1}", flush=True)
    sys.exit(0)

except Exception as e:
    print(f"\n❌ ERROR OCCURRED: {e}", flush=True)
    print(f"💾 Progress saved! Run again to resume from frame {current_frame_index + 1}", flush=True)
    sys.exit(1)

# Cleanup on completion
if os.path.exists(CHECKPOINT_FILE):
    os.remove(CHECKPOINT_FILE)

print("🎉 ALL FRAMES STYLIZED SUCCESSFULLY! 🎉", flush=True)
print(f"📁 Results saved in 'stylized/' folder", flush=True)