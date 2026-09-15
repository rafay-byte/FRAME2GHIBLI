<div align="center">

# 🎬 Frame2Ghibli

### Video-to-Ghibli Style Transfer Pipeline with Stable Diffusion & ControlNet

<p>
<img src="https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/Stable_Diffusion-FF4088?style=flat"/>
<img src="https://img.shields.io/badge/ControlNet-Canny-blue?style=flat"/>
<img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white"/>
<img src="https://img.shields.io/badge/OpenCV-5C3EE8?style=flat&logo=opencv&logoColor=white"/>
<img src="https://img.shields.io/badge/CUDA_Accelerated-76B900?style=flat&logo=nvidia&logoColor=white"/>
</p>

A generative AI video processing pipeline that transforms raw video footage into Studio Ghibli-style anime aesthetics<br/>using ControlNet edge conditioning, the MeinaMix diffusion checkpoint, and CUDA-accelerated post-processing.

</div>

---

## Overview

**Frame2Ghibli** is an end-to-end neural video stylization framework that converts standard video footage into high-fidelity anime and Studio Ghibli artistic renderings. Rather than applying unguided frame-by-frame diffusion (which produces extreme temporal flicker), Frame2Ghibli combines:

1. **Adaptive Canny Edge Conditioning** — Computes dynamic edge maps to preserve structural contours, facial features, and background geometry across frames.
2. **Stable Diffusion + ControlNet** — Runs `lllyasviel/sd-controlnet-canny` coupled with fine-tuned anime checkpoints (`MeinaMix_V11` / `anything-v5`) using the UniPCMultistepScheduler for fast, coherent sampling.
3. **CUDA-Accelerated Post-Processing (`STACK.py`)** — Hardware-accelerated watermark isolation, inpainting, and side-by-side comparison stacking.
4. **Video Reconstruction (`output.py`)** — Compiles stylized frames back into synchronized video streams.

## Sample Outputs

### Generated Ghibli-Style Output
<div align="center">
<img src="anime_output.png" alt="Ghibli Style Output" width="700"/>
</div>

### Frame Comparison & Stacking
<div align="center">
<img src="stacked_frames.png" alt="Stacked Comparison Frames" width="850"/>
</div>

## Pipeline Architecture

```
┌──────────────┐     ┌───────────────────────┐     ┌──────────────────────┐
│ Input Video  │────▶│    Frame Extractor    │────▶│ Adaptive Canny Edges │
│   (.mp4)     │     │      (OpenCV)         │     │ (Threshold Selection)│
└──────────────┘     └───────────────────────┘     └──────────┬───────────┘
                                                              │
                                                              ▼
┌──────────────┐     ┌───────────────────────┐     ┌──────────────────────┐
│ Output Video │◀────│   Reassembly/Encoding │◀────│ Stable Diffusion +   │
│ (Ghibli Style)     │ (CUDA Post-Process)   │     │ ControlNet (MeinaMix)│
└──────────────┘     └───────────────────────┘     └──────────────────────┘
```

## Features

- **ControlNet Canny Guidance** — Preserves geometric stability across successive frames via median-based adaptive edge thresholds.
- **MeinaMix & Anime Diffusion Checkpoints** — Integrated support for specialized anime latent diffusion models.
- **CUDA-Accelerated Post-Processing** — `CUDAWatermarkRemover` in `STACK.py` leverages GPU-accelerated morphological operations and inpainting.
- **RunwayML Integration** — `RUNWAYML.py` endpoint for cloud-based generative expansion.
- **Automated Frame Stacking** — Side-by-side comparison generators to evaluate stylization fidelity.

## Tech Stack

| Component | Technology |
|:---|:---|
| **Language** | Python 3.9+ |
| **Diffusion Framework** | HuggingFace Diffusers (`StableDiffusionControlNetPipeline`) |
| **Conditioning** | ControlNet Canny (`lllyasviel/sd-controlnet-canny`) |
| **Scheduler** | UniPCMultistepScheduler |
| **Base Models** | MeinaMix V11 / Anything-V5 |
| **Acceleration** | PyTorch CUDA & OpenCV CUDA (`cv2.cuda`) |
| **Video Processing** | OpenCV, FFmpeg, Pillow |

## Project Structure

```
FRAME2GHIBLI/
├── main.py                # Core ControlNet diffusion pipeline & adaptive Canny
├── output.py              # Frame sequencing and video reconstruction
├── STACK.py               # CUDA watermark removal & comparison stacking
├── RUNWAYML.py            # RunwayML API integration module
├── anime_output.png       # Verified sample Ghibli generation output
├── stacked_frames.png     # Frame-by-frame comparison result
├── MeinaMix_V11/          # Model checkpoint directory
├── input_frames/          # Extracted video frames
├── generated_frames/      # Diffused anime-style frames
├── output_frames/         # Post-processed frames
└── README.md
```

## Getting Started

### Prerequisites

- Python 3.9+
- NVIDIA GPU with 8GB+ VRAM (CUDA enabled)

### Installation

```bash
# Clone the repository
git clone https://github.com/rafay-byte/FRAME2GHIBLI.git
cd FRAME2GHIBLI

# Install dependencies
pip install torch torchvision diffusers transformers accelerate opencv-python Pillow tqdm
```

### Usage

```bash
# 1. Run the ControlNet diffusion pipeline on source frames
python main.py

# 2. Process frames and create side-by-side comparison stacks
python STACK.py

# 3. Reassemble stylized frames into final video
python output.py
```

## Contributing

Contributions, issues, and feature requests are welcome. Feel free to check the issues page or submit a pull request.

## License

This project is provided for research and educational purposes. Check individual model weights for applicable licenses.

---

<div align="center">
<sub>Developed by <a href="https://github.com/rafay-byte">Abdul Rafay Khalid</a> • BS AI Student @ PAF-IAST</sub>
</div>
