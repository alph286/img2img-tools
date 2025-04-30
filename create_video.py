import cv2
import os
import argparse
from pathlib import Path
import re
import subprocess
import shutil

def natural_sort_key(s):
    """
    Sort strings with numbers in a natural way
    For example: frame_1.png, frame_2.png, ..., frame_10.png
    Instead of: frame_1.png, frame_10.png, frame_2.png, ...
    """
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

def create_video_from_images(input_dir, output_path, fps=30, img_pattern="*.png", max_size_mb=None, quality=None):
    """
    Create a video from all images in the input directory
    
    Args:
        input_dir: Directory containing input images
        output_path: Path to save the output video
        fps: Frames per second for the output video
        img_pattern: Pattern to match image files (e.g., "*.png", "*.jpg")
        max_size_mb: Maximum size of the output video in MB
        quality: Video quality (0-100, lower means more compression)
    """
    # Get all image files matching the pattern
    image_files = []
    for ext in ["png", "jpg", "jpeg", "bmp", "webp"]:
        if img_pattern == f"*.{ext}" or img_pattern == "*.*":
            image_files.extend(list(Path(input_dir).glob(f"*.{ext}")))
            image_files.extend(list(Path(input_dir).glob(f"*.{ext.upper()}")))
    
    # Sort image files naturally
    image_files = sorted(image_files, key=lambda x: natural_sort_key(x.name))
    
    if not image_files:
        print(f"Error: No images found in {input_dir} matching pattern {img_pattern}")
        return
    
    print(f"Found {len(image_files)} images")
    
    # Read the first image to get dimensions
    first_image = cv2.imread(str(image_files[0]))
    if first_image is None:
        print(f"Error: Could not read image {image_files[0]}")
        return
    
    height, width, channels = first_image.shape
    print(f"Image dimensions: {width}x{height}")
    
    # Create a temporary file for initial video
    temp_output = output_path + ".temp.mp4"
    
    # Create video writer with H.264 codec for better compatibility
    fourcc = cv2.VideoWriter_fourcc(*'H264')
    
    # If H264 is not available, fall back to mp4v
    try:
        video_writer = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))
        if not video_writer.isOpened():
            raise Exception("Failed to open video writer with H264 codec")
    except:
        print("H264 codec not available, falling back to mp4v")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))
    
    if not video_writer.isOpened():
        print(f"Error: Could not create video writer for {temp_output}")
        return
    
    # Add each image to the video
    for i, img_path in enumerate(image_files):
        img = cv2.imread(str(img_path))
        
        if img is None:
            print(f"Warning: Could not read image {img_path}, skipping")
            continue
        
        # Resize image if dimensions don't match the first image
        if img.shape[0] != height or img.shape[1] != width:
            print(f"Resizing image {img_path} from {img.shape[1]}x{img.shape[0]} to {width}x{height}")
            img = cv2.resize(img, (width, height))
        
        # Add image to video
        video_writer.write(img)
        
        # Print progress every 100 frames or for the last frame
        if i % 100 == 0 or i == len(image_files) - 1:
            print(f"Added image {i+1}/{len(image_files)}: {img_path}")
    
    # Release video writer
    video_writer.release()
    
    # Check if FFmpeg is available
    ffmpeg_available = shutil.which("ffmpeg") is not None
    
    if max_size_mb and ffmpeg_available:
        # Calculate target bitrate based on max size
        # Formula: bitrate (kbps) = (target_size_kb * 8) / duration_seconds
        duration_seconds = len(image_files) / fps
        target_size_kb = max_size_mb * 1024
        target_bitrate_kbps = int((target_size_kb * 8) / duration_seconds)
        
        print(f"Compressing video to target size of {max_size_mb}MB...")
        print(f"Video duration: {duration_seconds:.2f} seconds")
        print(f"Target bitrate: {target_bitrate_kbps} kbps")
        
        # Use FFmpeg for better compression control
        cmd = [
            "ffmpeg", "-y",
            "-i", temp_output,
            "-c:v", "libx264",
            "-b:v", f"{target_bitrate_kbps}k",
            "-preset", "medium",
            "-pix_fmt", "yuv420p",  # Required for WhatsApp compatibility
            output_path
        ]
        
        # Add quality parameter if specified
        if quality is not None:
            # Convert quality (0-100) to CRF (0-51, lower is better)
            # 0 quality -> 51 CRF, 100 quality -> 0 CRF
            crf = max(0, min(51, int(51 - (quality / 100 * 51))))
            cmd[7:7] = ["-crf", str(crf)]
        
        print("Running FFmpeg compression...")
        subprocess.run(cmd, check=True)
        
        # Check final file size
        final_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"Final video size: {final_size_mb:.2f}MB")
        
        # Clean up temp file
        os.remove(temp_output)
    else:
        # If FFmpeg is not available or max_size not specified, just rename the temp file
        if os.path.exists(output_path):
            os.remove(output_path)
        os.rename(temp_output, output_path)
    
    print(f"Video created successfully: {output_path}")
    print(f"Video properties: {width}x{height}, {fps} FPS, {len(image_files)} frames")
    
    # Provide WhatsApp compatibility info
    print("\nWhatsApp compatibility tips:")
    print("- WhatsApp has a 16MB file size limit for videos")
    print("- If the video is still too large, try reducing the max_size_mb parameter")
    print("- For better quality at smaller sizes, try reducing the resolution or fps")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a video from images in a directory")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing input images")
    parser.add_argument("--output", type=str, required=True, help="Path to save the output video")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second for the output video")
    parser.add_argument("--pattern", type=str, default="*.png", help="Pattern to match image files (e.g., '*.png', '*.jpg')")
    parser.add_argument("--max_size_mb", type=float, default=None, help="Maximum size of the output video in MB (e.g., 16 for WhatsApp)")
    parser.add_argument("--quality", type=int, default=None, help="Video quality (0-100, lower means more compression)")
    
    args = parser.parse_args()
    
    create_video_from_images(args.input_dir, args.output, args.fps, args.pattern, args.max_size_mb, args.quality)