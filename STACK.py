import cv2
import numpy as np
from pathlib import Path


class CUDAWatermarkRemover:
    def __init__(self):
        self.use_cuda = cv2.cuda.getCudaEnabledDeviceCount() > 0
        if self.use_cuda:
            print(f"[✓] CUDA enabled with {cv2.cuda.getCudaEnabledDeviceCount()} device(s)")
        else:
            print("[!] CUDA not available, using CPU")

    def create_adaptive_mask_cuda(self, gpu_frame):
        """Create watermark mask using CUDA acceleration"""
        if self.use_cuda:
            # Convert to grayscale on GPU
            gpu_gray = cv2.cuda.cvtColor(gpu_frame, cv2.COLOR_BGR2GRAY)

            # Method 1: Threshold for bright areas (CUDA)
            _, gpu_mask1 = cv2.cuda.threshold(gpu_gray, 180, 255, cv2.THRESH_BINARY)

            # Method 2: Canny edge detection (CUDA)
            gpu_canny = cv2.cuda.Canny(gpu_gray, 50, 150)

            # Method 3: Another threshold for adaptive detection
            _, gpu_mask3 = cv2.cuda.threshold(gpu_gray, 200, 255, cv2.THRESH_BINARY)

            # Combine masks on GPU
            gpu_combined = cv2.cuda.bitwise_or(gpu_mask1, gpu_canny)
            gpu_combined = cv2.cuda.bitwise_or(gpu_combined, gpu_mask3)

            # Apply morphological operations (download to CPU for morphology)
            combined_cpu = gpu_combined.download()
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
            combined_cpu = cv2.morphologyEx(combined_cpu, cv2.MORPH_OPEN, kernel)
            combined_cpu = cv2.morphologyEx(combined_cpu, cv2.MORPH_CLOSE, kernel)

            # Upload back to GPU
            gpu_result = cv2.cuda_GpuMat()
            gpu_result.upload(combined_cpu)
            return gpu_result
        else:
            # Fallback to CPU
            frame = gpu_frame.download() if hasattr(gpu_frame, 'download') else gpu_frame
            return self.create_adaptive_mask_cpu(frame)

    @staticmethod
    def create_adaptive_mask_cpu(frame):
        """CPU fallback for mask creation"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Adaptive thresholding
        mask1 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY, 11, 2)
        mask1 = cv2.bitwise_not(mask1)

        # Edge detection
        edges = cv2.Canny(gray, 50, 150)

        # Brightness threshold
        _, mask3 = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)

        # Combine masks
        combined = cv2.bitwise_or(mask1, cv2.bitwise_or(edges, mask3))

        # Clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)

        return combined

    def inpaint_cuda(self, gpu_frame, gpu_mask):
        """CUDA-accelerated inpainting"""
        if self.use_cuda:
            # Download to CPU for inpainting (OpenCV inpaint not available on GPU)
            frame = gpu_frame.download()
            mask = gpu_mask.download() if hasattr(gpu_mask, 'download') else gpu_mask

            # Inpaint on CPU
            inpainted1 = cv2.inpaint(frame, mask, 3, cv2.INPAINT_TELEA)
            inpainted2 = cv2.inpaint(frame, mask, 3, cv2.INPAINT_NS)

            # Blend results
            result = cv2.addWeighted(inpainted1, 0.6, inpainted2, 0.4, 0)

            # Upload back to GPU for further processing
            gpu_result = cv2.cuda_GpuMat()
            gpu_result.upload(result)

            # Apply blur to watermark regions on GPU
            gpu_blurred = cv2.cuda.GaussianBlur(gpu_result, (3, 3), 0)

            # Simple blending (download for complex operations)
            blurred_cpu = gpu_blurred.download()
            mask_norm = mask.astype(np.float32) / 255.0
            mask_3ch = cv2.merge([mask_norm, mask_norm, mask_norm])

            # Blend on CPU
            final_result = result * (1 - mask_3ch) + blurred_cpu * mask_3ch

            return final_result.astype(np.uint8)
        else:
            # CPU fallback
            frame = gpu_frame if isinstance(gpu_frame, np.ndarray) else gpu_frame.download()
            mask = gpu_mask if isinstance(gpu_mask, np.ndarray) else gpu_mask.download()

            inpainted1 = cv2.inpaint(frame, mask, 3, cv2.INPAINT_TELEA)
            inpainted2 = cv2.inpaint(frame, mask, 3, cv2.INPAINT_NS)
            result = cv2.addWeighted(inpainted1, 0.6, inpainted2, 0.4, 0)

            # Apply selective blur
            blurred = cv2.GaussianBlur(result, (3, 3), 0)
            mask_norm = mask.astype(np.float32) / 255.0
            mask_3ch = cv2.merge([mask_norm, mask_norm, mask_norm])
            final_result = result * (1 - mask_3ch) + blurred * mask_3ch

            return final_result.astype(np.uint8)

    def remove_watermark(self, video_path, output_path="output_no_watermark.mp4"):
        """Main watermark removal function with CUDA acceleration"""
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Setup video writer
        fourcc = cv2.VideoWriter.fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        print(f"Processing {total_frames} frames with {'CUDA' if self.use_cuda else 'CPU'}...")

        # Initialize GPU matrices if CUDA available
        gpu_frame = None
        if self.use_cuda:
            gpu_frame = cv2.cuda_GpuMat()

        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if self.use_cuda and gpu_frame is not None:
                # Upload frame to GPU
                gpu_frame.upload(frame)

                # Process on GPU
                gpu_mask = self.create_adaptive_mask_cuda(gpu_frame)
                result = self.inpaint_cuda(gpu_frame, gpu_mask)
            else:
                # Process on CPU
                mask = self.create_adaptive_mask_cpu(frame)
                result = self.inpaint_cuda(frame, mask)

            out.write(result)

            frame_count += 1
            if frame_count % 30 == 0:
                print(f"Processed {frame_count}/{total_frames} frames")

        cap.release()
        out.release()
        print(f"[✓] Watermark removed! Output saved to: {output_path}")


def remove_watermark_cuda(video_path, output_path="output_no_watermark.mp4"):
    """Convenience function for CUDA watermark removal"""
    remover = CUDAWatermarkRemover()
    remover.remove_watermark(video_path, output_path)


def batch_process_cuda(input_dir, output_dir="cleaned_videos"):
    """Batch process videos with CUDA acceleration"""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    remover = CUDAWatermarkRemover()
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv'}

    for video_path in input_path.iterdir():
        if video_path.suffix.lower() in video_extensions:
            output_file = output_path / f"cleaned_{video_path.name}"
            print(f"Processing: {video_path.name}")
            remover.remove_watermark(str(video_path), str(output_file))


# Simplified CPU-only version for compatibility
def remove_watermark_simple(video_path, output_path="output_no_watermark.mp4"):
    """Simple CPU-only version for maximum compatibility"""

    def create_mask(frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, mask1 = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
        edges = cv2.Canny(gray, 50, 150)
        mask = cv2.bitwise_or(mask1, edges)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return mask

    cap = cv2.VideoCapture(str(video_path))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter.fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    print(f"Processing {total_frames} frames with CPU...")
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        mask = create_mask(frame)
        result = cv2.inpaint(frame, mask, 3, cv2.INPAINT_TELEA)
        out.write(result)

        frame_count += 1
        if frame_count % 30 == 0:
            print(f"Processed {frame_count}/{total_frames} frames")

    cap.release()
    out.release()
    print(f"[✓] Watermark removed! Output saved to: {output_path}")


# Usage examples:
if __name__ == "__main__":
    # Check CUDA availability
    cuda_available = cv2.cuda.getCudaEnabledDeviceCount() > 0
    print(f"OpenCV CUDA support: {cuda_available}")

    video_file = "When opportunity knocks and you re too busy to answer.mp4"

    if cuda_available:
        # Use CUDA version
        remove_watermark_cuda(video_file)
    else:
        # Use simple CPU version
        remove_watermark_simple(video_file)

    # Batch processing (uncomment to use)
    # batch_process_cuda("input_videos/", "output_videos/")