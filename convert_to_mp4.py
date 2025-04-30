import cv2
import os
import argparse
from pathlib import Path

def convert_to_mp4(input_video, output_path, fps=None, resize=None):
    """
    Convert a video file to MP4 format with H.264 codec
    
    Args:
        input_video: Path to the input video file
        output_path: Path to save the output MP4 video
        fps: Frames per second for the output video (None to keep original)
        resize: Tuple (width, height) to resize video (None to keep original size)
    """
    # Open the input video
    video = cv2.VideoCapture(input_video)
    
    # Check if video opened successfully
    if not video.isOpened():
        print(f"Error: Could not open video {input_video}")
        return
    
    # Get video properties
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    original_fps = video.get(cv2.CAP_PROP_FPS)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Use original fps if not specified
    if fps is None:
        fps = original_fps
    
    # Use original size if resize not specified
    if resize is not None:
        width, height = resize
    
    print(f"Input video: {width}x{height}, {original_fps} FPS, {total_frames} frames")
    print(f"Output video: {width}x{height}, {fps} FPS")
    
    # Create video writer with H.264 codec
    fourcc = cv2.VideoWriter_fourcc(*'H264')  # H.264 codec for MP4
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not video_writer.isOpened():
        print(f"Error: Could not create video writer for {output_path}")
        # Try alternative codec
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        print("Trying alternative codec (mp4v)...")
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if not video_writer.isOpened():
            print(f"Error: Could not create video writer with alternative codec")
            return
    
    # Process each frame
    frame_count = 0
    
    while True:
        # Read the next frame
        success, frame = video.read()
        
        if not success:
            break
        
        # Resize frame if needed
        if resize is not None:
            frame = cv2.resize(frame, (width, height))
        
        # Write frame to output video
        video_writer.write(frame)
        
        frame_count += 1
        
        # Print progress every 100 frames
        if frame_count % 100 == 0:
            print(f"Processed {frame_count}/{total_frames} frames ({frame_count/total_frames*100:.1f}%)")
    
    # Release resources
    video.release()
    video_writer.release()
    
    print(f"Conversion complete: {input_video} -> {output_path}")
    print(f"Converted {frame_count} frames")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert a video file to MP4 format")
    parser.add_argument("--input", type=str, required=True, help="Path to the input video file")
    parser.add_argument("--output", type=str, required=True, help="Path to save the output MP4 video")
    parser.add_argument("--fps", type=float, default=None, help="Frames per second for the output video")
    parser.add_argument("--width", type=int, default=None, help="Width of the output video")
    parser.add_argument("--height", type=int, default=None, help="Height of the output video")
    
    args = parser.parse_args()
    
    # Set resize dimensions if both width and height are provided
    resize = None
    if args.width is not None and args.height is not None:
        resize = (args.width, args.height)
    
    convert_to_mp4(args.input, args.output, args.fps, resize)