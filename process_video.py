import os
import cv2
import argparse
import torch
import re
import subprocess
import shutil
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from diffusers import StableDiffusionXLImg2ImgPipeline, DPMSolverMultistepScheduler

def natural_sort_key(s):
    """
    Sort strings with numbers in a natural way
    For example: frame_1.png, frame_2.png, ..., frame_10.png
    Instead of: frame_1.png, frame_10.png, frame_2.png, ...
    """
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

def clear_memory():
    """Clear CUDA memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

def extract_frames(video_path, output_dir, extract_all=True, max_frames=None):
    """
    Extract frames from a video file and save them as separate images
    
    Args:
        video_path: Path to the video file
        output_dir: Directory to save extracted frames
        extract_all: If True, extract all frames from the video
        max_frames: Maximum number of frames to extract (None for all frames)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Open the video file
    video = cv2.VideoCapture(video_path)
    
    # Check if video opened successfully
    if not video.isOpened():
        print(f"Errore: Impossibile aprire il video {video_path}")
        return False
    
    # Get video properties
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps > 0 else 0
    
    print(f"Info video: {total_frames} frames, {fps:.2f} FPS, {duration:.2f} secondi")
    
    # Calculate frame indices to extract
    if max_frames is not None and max_frames < total_frames:
        # Calcola l'intervallo per distribuire uniformemente i frame
        step = total_frames / max_frames
        frame_indices = [int(i * step) for i in range(max_frames)]
        print(f"Modalità test: estrazione di {max_frames} frames distribuiti uniformemente")
    else:
        frame_indices = list(range(total_frames))
    
    # Extract and save frames
    frames_extracted = 0
    
    for i, frame_idx in enumerate(tqdm(frame_indices, desc="Estrazione frames")):
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
                print(f"Salvato frame {i+1}/{len(frame_indices)} (posizione video: {frame_idx}) in {output_path}")
        else:
            print(f"Errore: Impossibile estrarre il frame alla posizione {frame_idx}")
    
    # Release the video file
    video.release()
    
    print(f"Estratti {frames_extracted} frames da {video_path}")
    return fps

def preprocess_image(image_path):
    """Preprocess an image for the model"""
    try:
        init_image = Image.open(image_path).convert("RGB")
        width, height = init_image.size
        
        # Resize to be compatible with SDXL (multiple of 8)
        target_width = (width // 8) * 8
        target_height = (height // 8) * 8
        
        if target_width != width or target_height != height:
            print(f"Ridimensionamento immagine a {target_width}x{target_height}")
            init_image = init_image.resize((target_width, target_height), Image.LANCZOS)
        
        return init_image
    except Exception as e:
        print(f"Errore nel preprocessamento dell'immagine {image_path}: {e}")
        return None

def process_images(input_dir, output_dir, prompt, model_path, strength=0.75, steps=20):
    """Process all images in the input directory using img2img"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create error directory
    error_dir = os.path.join(output_dir, "errors")
    os.makedirs(error_dir, exist_ok=True)
    
    # Get all image files
    image_files = sorted(list(Path(input_dir).glob("*.png")), key=lambda x: natural_sort_key(x.name))
    
    if not image_files:
        print(f"Nessuna immagine trovata in {input_dir}")
        return False
    
    print(f"Trovate {len(image_files)} immagini da processare")
    
    # Load model
    print(f"Caricamento modello da {model_path}...")
    
    # Check CUDA availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Controllo disponibilità CUDA...")
    print(f"Versione PyTorch: {torch.__version__}")
    print(f"CUDA disponibile secondo PyTorch: {torch.cuda.is_available()}")
    
    if device == "cuda":
        print(f"CUDA è disponibile. Utilizzo GPU: {torch.cuda.get_device_name(0)}")
        print(f"Conteggio dispositivi CUDA: {torch.cuda.device_count()}")
        print(f"Proprietà dispositivo CUDA: {torch.cuda.get_device_properties(0)}")
    
    try:
        # Load the pipeline
        pipe = StableDiffusionXLImg2ImgPipeline.from_single_file(
            model_path,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            use_safetensors=True,
            local_files_only=True,
        )
        
        # Force model to GPU
        if device == "cuda":
            pipe = pipe.to("cuda")
            print("Modello esplicitamente spostato su CUDA")
        
        # Use a simpler scheduler that's more compatible
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        print("Utilizzo scheduler DPMSolverMultistep per migliore compatibilità")
        
        # Enable memory efficient attention
        pipe.enable_attention_slicing(slice_size="max")
        pipe.enable_vae_slicing()  # Add VAE slicing to reduce memory usage
        
        # Disable safety checker if it exists
        if hasattr(pipe, "safety_checker") and pipe.safety_checker is not None:
            pipe.safety_checker = None
            print("Safety checker disabilitato per velocità")
        
        print(f"Modello caricato su {device}")
        
        # Clear memory after loading model
        clear_memory()
    except Exception as e:
        print(f"Errore nel caricamento del modello: {e}")
        return False
    
    # Process each image
    total_images = len(image_files)
    processed_count = 0
    
    for img_path in image_files:
        output_filename = f"processed_{img_path.name}"
        output_path = os.path.join(output_dir, output_filename)
        
        print(f"Elaborazione: {img_path} ({processed_count+1}/{total_images})")
        
        try:
            # Preprocess the image
            init_image = preprocess_image(img_path)
            
            if init_image is None:
                print(f"Saltando {img_path} a causa di un errore di preprocessamento")
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
                        print(f"Provo configurazione fallback {i}: strength={config['strength']}, steps={config['steps']}")
                    
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
                    print(f"Elaborazione completata: {img_path} -> {output_path}")
                    success = True
                    break
                    
                except RuntimeError as e:
                    if "cannot reshape tensor of 0 elements" in str(e) and i < len(fallback_configs) - 1:
                        print(f"Errore di reshape del tensore. Provo la prossima configurazione fallback...")
                        # Clear memory before trying again
                        clear_memory()
                    else:
                        raise
            
            if not success:
                print(f"Impossibile elaborare {img_path} dopo aver provato tutte le configurazioni fallback")
            
            # Clear memory after each image
            clear_memory()
            processed_count += 1
            
            # Print progress information
            print(f"Progresso: {processed_count}/{total_images} immagini elaborate ({processed_count/total_images*100:.1f}%)")
            
        except Exception as e:
            print(f"Errore nell'elaborazione di {img_path}: {e}")
            # Save a copy of the problematic image for inspection
            try:
                if 'init_image' in locals() and init_image:
                    init_image.save(os.path.join(error_dir, f"error_{img_path.name}"))
                    print(f"Immagine problematica salvata in {os.path.join(error_dir, f'error_{img_path.name}')}")
            except Exception as save_error:
                print(f"Impossibile salvare l'immagine di errore: {save_error}")
            
            processed_count += 1
    
    # Print completion message
    print(f"\nElaborazione completata! {processed_count}/{total_images} immagini elaborate.")
    print(f"Risultati salvati in: {output_dir}")
    return True

def create_video_from_images(input_dir, output_path, fps=30, max_size_mb=None, quality=None):
    """
    Create a video from all images in the input directory
    
    Args:
        input_dir: Directory containing input images
        output_path: Path to save the output video
        fps: Frames per second for the output video
        max_size_mb: Maximum size of the output video in MB
        quality: Video quality (0-100, lower means more compression)
    """
    # Get all image files
    image_files = sorted(list(Path(input_dir).glob("processed_*.png")), key=lambda x: natural_sort_key(x.name))
    
    if not image_files:
        print(f"Errore: Nessuna immagine trovata in {input_dir}")
        return False
    
    print(f"Trovate {len(image_files)} immagini")
    
    # Read the first image to get dimensions
    first_image = cv2.imread(str(image_files[0]))
    if first_image is None:
        print(f"Errore: Impossibile leggere l'immagine {image_files[0]}")
        return False
    
    height, width, channels = first_image.shape
    print(f"Dimensioni immagine: {width}x{height}")
    
    # Create a temporary file for initial video
    temp_output = output_path + ".temp.mp4"
    
    # Create video writer with H.264 codec for better compatibility
    fourcc = cv2.VideoWriter_fourcc(*'H264')
    
    # If H264 is not available, fall back to mp4v
    try:
        video_writer = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))
        if not video_writer.isOpened():
            raise Exception("Impossibile aprire il writer video con codec H264")
    except:
        print("Codec H264 non disponibile, utilizzo mp4v")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))
    
    if not video_writer.isOpened():
        print(f"Errore: Impossibile creare il writer video per {temp_output}")
        return False
    
    # Add each image to the video
    for i, img_path in enumerate(tqdm(image_files, desc="Creazione video")):
        img = cv2.imread(str(img_path))
        
        if img is None:
            print(f"Attenzione: Impossibile leggere l'immagine {img_path}, saltando")
            continue
        
        # Resize image if dimensions don't match the first image
        if img.shape[0] != height or img.shape[1] != width:
            print(f"Ridimensionamento immagine {img_path} da {img.shape[1]}x{img.shape[0]} a {width}x{height}")
            img = cv2.resize(img, (width, height))
        
        # Add image to video
        video_writer.write(img)
        
        # Print progress every 100 frames or for the last frame
        if i % 100 == 0 or i == len(image_files) - 1:
            print(f"Aggiunta immagine {i+1}/{len(image_files)}: {img_path}")
    
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
        
        print(f"Compressione video alla dimensione target di {max_size_mb}MB...")
        print(f"Durata video: {duration_seconds:.2f} secondi")
        print(f"Bitrate target: {target_bitrate_kbps} kbps")
        
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
        
        print("Esecuzione compressione FFmpeg...")
        subprocess.run(cmd, check=True)
        
        # Check final file size
        final_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"Dimensione finale video: {final_size_mb:.2f}MB")
        
        # Clean up temp file
        os.remove(temp_output)
    else:
        # If FFmpeg is not available or max_size not specified, just rename the temp file
        if os.path.exists(output_path):
            os.remove(output_path)
        os.rename(temp_output, output_path)
    
    print(f"Video creato con successo: {output_path}")
    print(f"Proprietà video: {width}x{height}, {fps} FPS, {len(image_files)} frames")
    return True

def process_video(video_path, output_dir, prompt, model_path, strength=0.75, steps=20, max_size_mb=None, quality=None, max_frames=None):
    """Process a video through the entire pipeline"""
    # Create directories
    frames_dir = os.path.join(output_dir, "frames")
    temp_frames_dir = os.path.join(output_dir, "tempframes")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(temp_frames_dir, exist_ok=True)
    
    # Output video path
    output_video = os.path.join(output_dir, "output_video.mp4")
    
    print("\n=== FASE 1: ESTRAZIONE FRAMES ===")
    fps = extract_frames(video_path, frames_dir, max_frames=max_frames)
    if not fps:
        print("Errore nell'estrazione dei frames. Uscita.")
        return False
    
    print("\n=== FASE 2: ELABORAZIONE IMMAGINI ===")
    success = process_images(frames_dir, temp_frames_dir, prompt, model_path, strength, steps)
    if not success:
        print("Errore nell'elaborazione delle immagini. Uscita.")
        return False
    
    print("\n=== FASE 3: CREAZIONE VIDEO ===")
    success = create_video_from_images(temp_frames_dir, output_video, fps, max_size_mb, quality)
    if not success:
        print("Errore nella creazione del video. Uscita.")
        return False
    
    print("\n=== ELABORAZIONE COMPLETATA ===")
    print(f"Video originale: {video_path}")
    print(f"Video elaborato: {output_video}")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Elabora un video con Stable Diffusion")
    parser.add_argument("--video", type=str, required=True, help="Percorso al file video di input")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory per salvare l'output")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt per guidare la generazione")
    parser.add_argument("--model_path", type=str, required=True, help="Percorso al modello Stable Diffusion")
    parser.add_argument("--strength", type=float, default=0.75, help="Intensità della trasformazione (0.0-1.0)")
    parser.add_argument("--steps", type=int, default=20, help="Numero di passi di inferenza")
    parser.add_argument("--max_size_mb", type=float, default=None, help="Dimensione massima del file video in MB")
    parser.add_argument("--quality", type=int, default=None, help="Qualità video (0-100, più basso = più compressione)")
    parser.add_argument("--max_frames", type=int, default=None, help="Numero massimo di frames da elaborare (modalità test)")
    
    args = parser.parse_args()
    
    process_video(
        args.video,
        args.output_dir,
        args.prompt,
        args.model_path,
        args.strength,
        args.steps,
        args.max_size_mb,
        args.quality,
        args.max_frames
    )