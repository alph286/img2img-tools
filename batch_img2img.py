import os
import argparse
from pathlib import Path
import torch
import gc
from diffusers import StableDiffusionXLImg2ImgPipeline
from PIL import Image

# Add this function at the top of your script
def clear_memory():
    """Clear CUDA memory to prevent out-of-memory errors"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    gc.collect()

# Add this function after the clear_memory function
# Modify the preprocess_image function
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
        print(f"Resized image to {new_width}x{new_height} (shortest side 512px)")
        
        return image
    except Exception as e:
        print(f"Error preprocessing image {image_path}: {e}")
        return None

def process_images(input_dir, output_dir, prompt, model_path="sd_xl_turbo_1.0_fp16.safetensors", strength=0.75, steps=20):
    """
    Process all images in the input directory using img2img
    
    Args:
        input_dir: Directory containing input images
        output_dir: Directory to save processed images
        prompt: Text prompt for img2img
        model_path: Path to the local model file
        strength: How much to transform the image (0-1)
        steps: Number of sampling steps
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all image files (common formats)
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(list(Path(input_dir).glob(f'*{ext}')))
        image_files.extend(list(Path(input_dir).glob(f'*{ext.upper()}')))
    
    print(f"Found {len(image_files)} images to process")
    
    # Load the model from local file
    print(f"Loading model from {model_path}...")
    
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
                sample_param = next(pipe.parameters())
                if sample_param.device.type != "cuda":
                    print("Warning: Model not properly moved to CUDA. Trying alternative method...")
                    
                    # Alternative method to move model to CUDA
                    for name, param in pipe.named_parameters():
                        param.data = param.data.to("cuda")
                    
                    # Check again
                    sample_param = next(pipe.parameters())
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
    
    # Process each image
    total_images = len(image_files)
    processed_count = 0
    
    for img_path in image_files:
        output_filename = f"processed_{img_path.name}"
        output_path = os.path.join(output_dir, output_filename)
        
        print(f"Processing: {img_path} ({processed_count+1}/{total_images})")
        
        try:
            # Preprocess the image
            init_image = preprocess_image(img_path)
            
            if init_image is None:
                print(f"Skipping {img_path} due to preprocessing error")
                processed_count += 1
                continue
            
            # Run img2img with more robust error handling
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
                    if i > 0:
                        print(f"Trying fallback configuration {i}: strength={config['strength']}, steps={config['steps']}")
                    
                    with torch.no_grad():
                        result = pipe(
                            prompt=prompt,
                            image=init_image,
                            strength=config["strength"],
                            num_inference_steps=config["steps"],
                            guidance_scale=config["guidance_scale"]
                        ).images[0]
                    
                    # Save the result
                    result.save(output_path)
                    print(f"Successfully processed: {img_path} -> {output_path}")
                    success = True
                    break
                    
                except RuntimeError as e:
                    if "cannot reshape tensor of 0 elements" in str(e) and i < len(fallback_configs) - 1:
                        print(f"Encountered tensor reshape error. Trying next fallback configuration...")
                        # Clear memory before trying again
                        clear_memory()
                    else:
                        raise
            
            if not success:
                print(f"Failed to process {img_path} after trying all fallback configurations")
            
            # Clear memory after each image
            clear_memory()
            processed_count += 1
            
            # Print progress information
            print(f"Progress: {processed_count}/{total_images} images processed ({processed_count/total_images*100:.1f}%)")
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            # Save a copy of the problematic image for inspection
            try:
                error_dir = os.path.join(output_dir, "errors")
                os.makedirs(error_dir, exist_ok=True)
                if 'init_image' in locals() and init_image:
                    init_image.save(os.path.join(error_dir, f"error_{img_path.name}"))
                    print(f"Saved problematic image to {os.path.join(error_dir, f'error_{img_path.name}')}")
            except Exception as save_error:
                print(f"Could not save error image: {save_error}")
            
            processed_count += 1
    
    # Print completion message
    print(f"\nProcessing complete! {processed_count}/{total_images} images processed.")
    print(f"Results saved to: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch process images with img2img")
    parser.add_argument("--input_dir", type=str, required=True, help="Input directory with images")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for processed images")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for img2img")
    parser.add_argument("--model_path", type=str, default="c:\\Users\\Kratos\\Desktop\\img2img\\sd_xl_turbo_1.0_fp16.safetensors", 
                        help="Path to the local model file")
    parser.add_argument("--strength", type=float, default=0.75, help="Transformation strength (0-1)")
    parser.add_argument("--steps", type=int, default=1, help="Number of inference steps (1-2 recommended for Turbo)")
    
    args = parser.parse_args()
    
    process_images(
        args.input_dir,
        args.output_dir,
        args.prompt,
        args.model_path,
        args.strength,
        args.steps
    )