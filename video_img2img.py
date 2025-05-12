import os
import argparse
import subprocess
import shutil
from pathlib import Path
import torch
import gc
import cv2
import numpy as np
from diffusers import StableDiffusionXLImg2ImgPipeline
from PIL import Image
from tqdm import tqdm

# Add this function at the top of your script
def clear_memory():
    """Clear CUDA memory to prevent out-of-memory errors"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    gc.collect()

def preprocess_image(image_path):
    """
    Preprocess image to ensure it's compatible with the model
    """
    try:
        # Load image
        image = Image.open(str(image_path)).convert("RGB")
        
        # Resize to 512 pixels on the shortest side while maintaining aspect ratio
        width, height = image.size
        
        if width < height:
            # Width is the shorter side
            new_width = 512
            new_height = int(height * (512 / width))
        else:
            # Height is the shorter side or they're equal
            new_height = 512
            new_width = int(width * (512 / height))
            
        # Resize image to target dimensions
        image = image.resize((new_width, new_height), Image.LANCZOS)
        
        return image
    except Exception as e:
        print(f"Error preprocessing image {image_path}: {e}")
        return None

def upscale_image(image, target_height=1080):
    """
    Upscale image to target height while maintaining aspect ratio
    """
    try:
        width, height = image.size
        
        # Calculate new dimensions
        scale_factor = target_height / height
        new_width = int(width * scale_factor)
        new_height = target_height
        
        # Resize image to target dimensions
        upscaled_image = image.resize((new_width, new_height), Image.LANCZOS)
        
        return upscaled_image
    except Exception as e:
        print(f"Error upscaling image: {e}")
        return image  # Return original image if upscaling fails

def extract_frames(video_path, output_dir):
    """
    Extract frames from video
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get video info
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Extracting {frame_count} frames from video at {fps} FPS...")
    
    # Extract frames
    success, frame = video.read()
    frame_number = 0
    
    with tqdm(total=frame_count) as pbar:
        while success:
            frame_path = os.path.join(output_dir, f"frame_{frame_number:06d}.png")
            cv2.imwrite(frame_path, frame)
            success, frame = video.read()
            frame_number += 1
            pbar.update(1)
    
    video.release()
    
    print(f"Extracted {frame_number} frames to {output_dir}")
    return fps, frame_number

def reassemble_video(frames_dir, output_path, fps, audio_path=None):
    """
    Reassemble frames into video
    """
    # Get first frame to determine dimensions
    first_frame = sorted(os.listdir(frames_dir))[0]
    first_frame_path = os.path.join(frames_dir, first_frame)
    img = cv2.imread(first_frame_path)
    height, width, _ = img.shape
    
    print(f"Reassembling video with dimensions {width}x{height} at {fps} FPS...")
    
    # Create temporary video without audio
    temp_output = output_path + ".temp.mp4"
    
    # Use ffmpeg to create video from frames
    frame_pattern = os.path.join(frames_dir, "frame_%06d.png")
    
    # Create video from frames
    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", frame_pattern,
        "-c:v", "libx264",
        "-preset", "slow",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        temp_output
    ]
    
    subprocess.run(ffmpeg_cmd, check=True)
    
    # If original video has audio, add it to the output
    if audio_path:
        final_cmd = [
            "ffmpeg", "-y",
            "-i", temp_output,
            "-i", audio_path,
            "-c:v", "copy",
            "-c:a", "aac",
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-shortest",
            output_path
        ]
        subprocess.run(final_cmd, check=True)
        os.remove(temp_output)
    else:
        # If no audio, just rename the temp file
        os.rename(temp_output, output_path)
    
    print(f"Video reassembled and saved to {output_path}")

def extract_audio(video_path, output_path):
    """
    Extract audio from video
    """
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vn",
        "-acodec", "copy",
        output_path
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print(f"Audio extracted to {output_path}")
        return output_path
    except subprocess.CalledProcessError:
        print("No audio stream found or error extracting audio")
        return None

def process_video(input_video, output_video, prompt, model_path="sd_xl_turbo_1.0_fp16.safetensors", 
                 strength=0.75, steps=20, target_height=1080, use_realesrgan=True, 
                 max_frames=None, max_size_mb=None, quality=70):
    """
    Process video with img2img
    
    Args:
        input_video: Path to input video
        output_video: Path to output video
        prompt: Text prompt for img2img
        model_path: Path to the local model file
        strength: How much to transform the image (0-1)
        steps: Number of sampling steps
        target_height: Target height for upscaling (default: 1080)
        use_realesrgan: Whether to use Real-ESRGAN for upscaling
        max_frames: Maximum number of frames to process (for testing)
        max_size_mb: Maximum size of output video in MB
        quality: Output video quality (0-100)
    """
    # Create temporary directories
    temp_dir = os.path.join(os.path.dirname(output_video), "temp_video_processing")
    frames_dir = os.path.join(temp_dir, "frames")
    processed_frames_dir = os.path.join(temp_dir, "processed_frames")
    upscaled_frames_dir = os.path.join(temp_dir, "upscaled_frames")
    
    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(processed_frames_dir, exist_ok=True)
    os.makedirs(upscaled_frames_dir, exist_ok=True)
    
    try:
        # Step 1: Extract frames
        fps, frame_count = extract_frames(input_video, frames_dir)
        
        # Limita il numero di frame se specificato
        if max_frames is not None and max_frames > 0:
            frame_count = min(frame_count, max_frames)
            print(f"Limitando l'elaborazione a {max_frames} frames")
        
        # Step 2: Extract audio
        audio_path = os.path.join(temp_dir, "audio.aac")
        audio_path = extract_audio(input_video, audio_path)
        
        # Step 3: Process frames with img2img
        # Force CUDA device if available - improved detection
        print("Checking CUDA availability...")
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available according to PyTorch: {torch.cuda.is_available()}")
        
        device = "cpu"  # Default to CPU
        
        # Improved CUDA detection and initialization
        if torch.cuda.is_available():
            try:
                # Reset CUDA before testing
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                
                # Test CUDA availability more thoroughly
                test_tensor = torch.zeros(1).cuda()
                # Try a small computation to verify CUDA is working
                test_result = test_tensor + 1
                
                # Force synchronization to ensure CUDA operations complete
                torch.cuda.synchronize()
                
                # Check if the result is correct
                if test_result.item() == 1:
                    device = "cuda"
                    print(f"CUDA is available and working. Using GPU: {torch.cuda.get_device_name()}")
                    print(f"CUDA device count: {torch.cuda.device_count()}")
                    print(f"CUDA memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
                    print(f"CUDA memory reserved: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
                    
                    # Try to initialize cuDNN to ensure it's working
                    torch.backends.cudnn.enabled = True
                    
                    # Force initialize CUDA context
                    dummy = torch.ones(1).cuda()
                    del dummy
                    torch.cuda.synchronize()
                else:
                    print("CUDA test computation failed. Falling back to CPU.")
                
                # Clean up test tensors
                del test_tensor, test_result
                torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"CUDA reported as available but failed in testing: {e}")
                print("WARNING: Falling back to CPU (this will be very slow)")
                
                # Try to reinitialize CUDA
                try:
                    print("Attempting to reinitialize CUDA...")
                    torch.cuda.empty_cache()
                    torch.cuda.reset_max_memory_allocated()
                    torch.cuda.reset_max_memory_cached()
                    
                    # Try a different CUDA initialization approach
                    dummy = torch.cuda.FloatTensor(1)
                    del dummy
                    torch.cuda.synchronize()
                    
                    device = "cuda"
                    print("CUDA reinitialized successfully!")
                except Exception as reinit_error:
                    print(f"CUDA reinitialization failed: {reinit_error}")
                    device = "cpu"
        else:
            print("WARNING: CUDA not available. Using CPU (this will be very slow)")
            print("To enable CUDA, you may need to:")
            print("1. Install PyTorch with CUDA support: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
            print("2. Make sure your NVIDIA drivers are up to date")
            print("3. Check if your GPU supports CUDA")
        
        # Disable symlinks warning and set environment variable to prevent symlink creation
        os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
        os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
        
        # Performance optimizations
        if device == "cuda":
            try:
                torch.backends.cudnn.benchmark = True
                if hasattr(torch.backends.cuda, "matmul"):
                    torch.backends.cuda.matmul.allow_tf32 = True
                if hasattr(torch.backends, "cudnn"):
                    torch.backends.cudnn.allow_tf32 = True
                # Set a reasonable memory limit to prevent OOM errors
                if hasattr(torch.cuda, "set_per_process_memory_fraction"):
                    torch.cuda.set_per_process_memory_fraction(0.8)  # Use 80% of available VRAM
            except Exception as e:
                print(f"Warning: Could not set all CUDA optimizations: {e}")
        
        try:
            # For SDXL Turbo, we need to use the SDXL pipeline
            print("Loading pipeline components...")
            
            # Use a simpler approach without trying to load VAE separately
            pipe = StableDiffusionXLImg2ImgPipeline.from_single_file(
                model_path,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                use_safetensors=True,
                local_files_only=True,
            )
            
            # Force model to GPU with better error handling
            if device == "cuda":
                try:
                    # Clear CUDA cache before loading model
                    torch.cuda.empty_cache()
                    
                    # Try to move model to CUDA
                    pipe = pipe.to("cuda")
                    
                    # Verify model is actually on CUDA
                    # Check a core component like unet for its device
                    sample_param = next(pipe.unet.parameters())
                    if sample_param.device.type != "cuda":
                        print("Warning: Model not properly moved to CUDA. Trying alternative method...")
                        
                        # Alternative method to move model to CUDA
                        for name, param in pipe.named_parameters():
                            param.data = param.data.to("cuda")
                        
                        # Check again
                        sample_param = next(pipe.unet.parameters())
                        if sample_param.device.type == "cuda":
                            print("Successfully moved model to CUDA using alternative method")
                        else:
                            print("WARNING: Could not move model to CUDA despite multiple attempts")
                            print("Trying one last approach...")
                            
                            # Last resort: reload model directly to CUDA
                            del pipe
                            torch.cuda.empty_cache()
                            torch.cuda.synchronize()
                            
                            pipe = StableDiffusionXLImg2ImgPipeline.from_single_file(
                                model_path,
                                torch_dtype=torch.float16,
                                use_safetensors=True,
                                local_files_only=True,
                                device_map="cuda"
                            )
                    else:
                        print("Model successfully moved to CUDA")
                    
                    # Force synchronization to ensure model is loaded
                    torch.cuda.synchronize()
                    
                except Exception as cuda_err:
                    print(f"Error moving model to CUDA: {cuda_err}")
                    print("Falling back to CPU")
                    device = "cpu"
                    pipe = pipe.to("cpu")
                print("Model explicitly moved to CUDA")
            
            # Use a simpler scheduler that's more compatible
            from diffusers import DPMSolverMultistepScheduler
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
            print("Using DPMSolverMultistep scheduler for better compatibility")
            
            # Enable memory efficient attention
            pipe.enable_attention_slicing(slice_size="max")
            pipe.enable_vae_slicing()  # Add VAE slicing to reduce memory usage
            
            # Disable safety checker if it exists
            if hasattr(pipe, "safety_checker") and pipe.safety_checker is not None:
                pipe.safety_checker = None
                print("Safety checker disabled for speed")
            
            print(f"Model loaded on {device}")
            
            # Clear memory after loading model
            clear_memory()
        except Exception as e:
            print(f"Error loading model with from_single_file: {e}")
            print("Trying alternative loading method...")
            
            # Alternative loading method for local models
            try:
                from diffusers import AutoPipelineForImage2Image
                pipe = AutoPipelineForImage2Image.from_single_file(
                    model_path,
                    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                    use_safetensors=True,
                    local_files_only=True
                )
                pipe = pipe.to(device)
                print(f"Model loaded with AutoPipelineForImage2Image on {device}")
            except Exception as e2:
                print(f"Error with alternative loading method: {e2}")
                raise RuntimeError("Failed to load model. Please run as administrator or enable Developer Mode in Windows.")
        
        # Process each frame
        all_frame_files = sorted([f for f in os.listdir(frames_dir) if f.startswith("frame_")])
        # Apply max_frames limit to the list of files to be processed
        if max_frames is not None and max_frames > 0:
            frame_files = all_frame_files[:max_frames]
        else:
            frame_files = all_frame_files
        total_frames = len(frame_files)
        
        print(f"Processing {total_frames} frames with img2img...")
        
        # Try to load Real-ESRGAN for upscaling
        try:
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer
            
            # Initialize upscaler
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
            upscaler = RealESRGANer(
                scale=2,
                model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth',
                model=model,
                tile=400,
                tile_pad=10,
                pre_pad=0,
                half=device == 'cuda'
            )
            print("Real-ESRGAN upscaler loaded successfully")
            use_realesrgan = True
        except ImportError:
            print("Real-ESRGAN not available. Using standard upscaling.")
            use_realesrgan = False
        
        with tqdm(total=total_frames) as pbar:
            for frame_file in frame_files:
                frame_path = os.path.join(frames_dir, frame_file)
                processed_path = os.path.join(processed_frames_dir, frame_file)
                upscaled_path = os.path.join(upscaled_frames_dir, frame_file)
                
                try:
                    # Step 2: Preprocess frame (resize to 512px on shortest side)
                    init_image = preprocess_image(frame_path)
                    
                    if init_image is None:
                        print(f"Skipping {frame_path} due to preprocessing error")
                        # Copy original frame to maintain sequence
                        shutil.copy(frame_path, processed_path)
                        shutil.copy(frame_path, upscaled_path)
                        pbar.update(1)
                        continue
                    
                    # Step 3: Apply img2img
                    success = False
                    
                    # Try with progressively more conservative parameters until success
                    fallback_configs = [
                        {"strength": strength, "steps": max(1, steps), "guidance_scale": 0.0},
                        {"strength": min(0.8, strength + 0.1), "steps": max(2, steps), "guidance_scale": 1.0},
                        {"strength": 0.4, "steps": 4, "guidance_scale": 1.0},
                        {"strength": 0.2, "steps": 5, "guidance_scale": 1.5}
                    ]
                    
                    for i, config in enumerate(fallback_configs):
                        try:
                            with torch.no_grad():
                                result = pipe(
                                    prompt=prompt,
                                    image=init_image,
                                    strength=config["strength"],
                                    num_inference_steps=config["steps"],
                                    guidance_scale=config["guidance_scale"]
                                ).images[0]
                            
                            # Save the processed frame
                            result.save(processed_path)
                            success = True
                            break
                                
                        except RuntimeError as e:
                            if "cannot reshape tensor of 0 elements" in str(e) and i < len(fallback_configs) - 1:
                                # Clear memory before trying again
                                clear_memory()
                            else:
                                raise
                    
                    if not success:
                        print(f"Failed to process {frame_path} after trying all fallback configurations")
                        # Copy original frame to maintain sequence
                        shutil.copy(frame_path, processed_path)
                        shutil.copy(frame_path, upscaled_path)
                        pbar.update(1)
                        continue
                    
                    # Step 4: Upscale to 1080p height
                    if use_realesrgan:
                        # Use Real-ESRGAN for better upscaling
                        img = cv2.imread(processed_path, cv2.IMREAD_UNCHANGED)
                        h, w = img.shape[:2]
                        
                        # Calculate target size for 1080p height
                        target_h = 1080
                        target_w = int(w * (target_h / h))
                        
                        # First use Real-ESRGAN to upscale by 2x
                        output, _ = upscaler.enhance(img)
                        
                        # Then resize to exact target dimensions
                        output = cv2.resize(output, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
                        
                        # Save upscaled image
                        cv2.imwrite(upscaled_path, output)
                    else:
                        # Use PIL for standard upscaling
                        processed_img = Image.open(processed_path)
                        upscaled_img = upscale_image(processed_img, target_height=target_height)
                        upscaled_img.save(upscaled_path)
                    
                    # Clear memory after processing each frame
                    clear_memory()
                    pbar.update(1)
                    
                except Exception as e:
                    print(f"Error processing frame {frame_path}: {e}")
                    # Copy original frame to maintain sequence
                    shutil.copy(frame_path, processed_path)
                    shutil.copy(frame_path, upscaled_path)
                    pbar.update(1)
        
        # Step 5: Reassemble video
        reassemble_video(upscaled_frames_dir, output_video, fps, audio_path)
        
        print(f"Video processing complete! Output saved to {output_video}")
        
    finally:
        # Clean up temporary files
        if os.path.exists(temp_dir):
            print(f"Cleaning up temporary files in {temp_dir}...")
            shutil.rmtree(temp_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process video with img2img")
    parser.add_argument("--input_video", type=str, required=True, help="Input video file")
    parser.add_argument("--output_video", type=str, required=True, help="Output video file")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for img2img")
    parser.add_argument("--model_path", type=str, default="sd_xl_turbo_1.0_fp16.safetensors", 
                        help="Path to the local model file")
    parser.add_argument("--strength", type=float, default=0.75, help="Transformation strength (0-1)")
    parser.add_argument("--steps", type=int, default=1, help="Number of inference steps (1-2 recommended for Turbo)")
    
    args = parser.parse_args()
    
    process_video(
        args.input_video,
        args.output_video,
        args.prompt,
        args.model_path,
        args.strength,
        args.steps
    )