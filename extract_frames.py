import cv2
import os
import argparse
from pathlib import Path

def extract_frames(video_path, output_dir, extract_all=False, num_frames=20, from_start=True):
    """
    Extract frames from a video file and save them as separate images
    
    Args:
        video_path: Path to the video file
        output_dir: Directory to save extracted frames
        extract_all: If True, extract all frames from the video
        num_frames: Number of frames to extract if extract_all is False
        from_start: If True, extract the first num_frames frames
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Open the video file
    video = cv2.VideoCapture(video_path)
    
    # Check if video opened successfully
    if not video.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    # Get video properties
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps > 0 else 0
    
    print(f"Video info: {total_frames} frames, {fps:.2f} FPS, {duration:.2f} seconds")
    
    # Calculate frame indices to extract
    if extract_all:
        print(f"Extracting all {total_frames} frames...")
        frame_indices = list(range(total_frames))
    elif from_start:
        # Extract the first num_frames frames
        frame_indices = list(range(min(num_frames, total_frames)))
    else:
        # Calculate step size to get evenly distributed frames
        if total_frames <= num_frames:
            frame_indices = list(range(total_frames))
        else:
            step = total_frames / num_frames
            frame_indices = [int(i * step) for i in range(num_frames)]
    
    # Extract and save frames
    frames_extracted = 0
    
    for i, frame_idx in enumerate(frame_indices):
        # Set video position to the desired frame
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        
        # Read the frame
        success, frame = video.read()
        
        if success:
            # Save the frame as an image
            output_path = os.path.join(output_dir, f"frame_{i:05d}.png")
            cv2.imwrite(output_path, frame)
            frames_extracted += 1
            
            # Print progress every 100 frames or for the last frame
            if i % 100 == 0 or i == len(frame_indices) - 1:
                print(f"Saved frame {i+1}/{len(frame_indices)} (video position: {frame_idx}) to {output_path}")
        else:
            print(f"Error: Failed to extract frame at position {frame_idx}")
    
    # Release the video file
    video.release()
    
    print(f"Extracted {frames_extracted} frames from {video_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames from a video file")
    parser.add_argument("--video", type=str, required=True, help="Path to the video file")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save extracted frames")
    parser.add_argument("--extract_all", action="store_true", help="Extract all frames from the video")
    parser.add_argument("--num_frames", type=int, default=20, help="Number of frames to extract if not extracting all")
    parser.add_argument("--from_start", action="store_true", default=True, 
                        help="Extract frames from the start of the video")
    
    args = parser.parse_args()
    
    extract_frames(args.video, args.output_dir, args.extract_all, args.num_frames, args.from_start)