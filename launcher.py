import os
import sys
import subprocess
import pkg_resources
import shutil
import requests
from tqdm import tqdm
import platform

def check_ffmpeg():
    """Verifica se FFmpeg è installato nel sistema."""
    try:
        subprocess.run(
            ["ffmpeg", "-version"], 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            check=True
        )
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        return False

def install_ffmpeg():
    """Installa FFmpeg in base al sistema operativo."""
    system = platform.system()
    print("FFmpeg non trovato. Installazione in corso...")
    
    if system == "Windows":
        print("Per Windows, è necessario installare FFmpeg manualmente.")
        print("1. Scarica FFmpeg da: https://ffmpeg.org/download.html")
        print("2. Estrai il contenuto e aggiungi la cartella bin al PATH di sistema")
        input("Premi Invio dopo aver installato FFmpeg per continuare...")
    elif system == "Darwin":  # macOS
        try:
            subprocess.run(["brew", "install", "ffmpeg"], check=True)
        except (subprocess.SubprocessError, FileNotFoundError):
            print("Installazione tramite Homebrew fallita.")
            print("Installa FFmpeg manualmente da: https://ffmpeg.org/download.html")
            input("Premi Invio dopo aver installato FFmpeg per continuare...")
    else:  # Linux
        try:
            subprocess.run(["apt-get", "update"], check=True)
            subprocess.run(["apt-get", "install", "-y", "ffmpeg"], check=True)
        except (subprocess.SubprocessError, FileNotFoundError):
            print("Installazione tramite apt fallita.")
            print("Installa FFmpeg manualmente usando il gestore pacchetti del tuo sistema.")
            input("Premi Invio dopo aver installato FFmpeg per continuare...")

def check_and_install_dependencies():
    """Controlla e installa le dipendenze Python necessarie."""
    required_packages = {
        "torch": "torch>=2.0.0",
        "diffusers": "diffusers>=0.19.0",
        "opencv-python": "opencv-python>=4.5.0",
        "transformers": "transformers>=4.30.0",
        "accelerate": "accelerate>=0.20.0",
        "safetensors": "safetensors>=0.3.0",
        "tqdm": "tqdm>=4.65.0",
        "requests": "requests>=2.28.0"
    }
    
    missing_packages = []
    
    # Verifica quali pacchetti sono mancanti
    for package, requirement in required_packages.items():
        try:
            pkg_resources.require(requirement)
        except (pkg_resources.DistributionNotFound, pkg_resources.VersionConflict):
            missing_packages.append(requirement)
    
    # Installa i pacchetti mancanti
    if missing_packages:
        print(f"Installazione delle dipendenze mancanti: {', '.join(missing_packages)}")
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing_packages)
        print("Tutte le dipendenze sono state installate con successo!")
    else:
        print("Tutte le dipendenze Python sono già installate.")
    
    # Verifica FFmpeg
    if not check_ffmpeg():
        install_ffmpeg()
        return False
    return True

def download_model(url, destination):
    """Scarica un file con barra di progresso."""
    if not os.path.exists(os.path.dirname(destination)):
        os.makedirs(os.path.dirname(destination))
    
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(destination, 'wb') as file, tqdm(
        desc=os.path.basename(destination),
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)

def main():
    print("=== Launcher di img2img-tools ===")
    
    # Controlla e installa le dipendenze
    print("\nVerifica delle dipendenze in corso...")
    ffmpeg_installed = check_and_install_dependencies()
    
    # Crea la cartella checkpoints se non esiste
    checkpoints_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)
    
    # Percorso del modello SDXL Turbo
    model_path = os.path.join(checkpoints_dir, "sd_xl_turbo_1.0_fp16.safetensors")
    
    # Chiedi all'utente se vuole scaricare il modello SDXL Turbo
    if not os.path.exists(model_path):
        download_choice = input("\nVuoi scaricare il modello SDXL Turbo? (sì/no): ").lower()
        
        if download_choice in ["sì", "si", "s", "yes", "y"]:
            print("\nScaricamento del modello SDXL Turbo in corso...")
            # URL per il modello SDXL Turbo (safetensors)
            model_url = "https://huggingface.co/stabilityai/sdxl-turbo/resolve/main/sd_xl_turbo_1.0_fp16.safetensors"
            download_model(model_url, model_path)
            print(f"\nModello scaricato con successo in: {model_path}")
        else:
            print("\nScaricamento del modello saltato.")
    else:
        print(f"\nIl modello SDXL Turbo è già presente in: {model_path}")
    
    # Avvia l'applicazione principale solo se le dipendenze sono soddisfatte e FFmpeg è installato
    if ffmpeg_installed:
        app_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app.py")
        if os.path.exists(app_path):
            print("\nAvvio dell'applicazione img2img-tools...")
            try:
                subprocess.run([sys.executable, app_path], check=True)
            except subprocess.SubprocessError:
                print("\nErrore durante l'avvio dell'applicazione.")
        else:
            print("\nImpossibile trovare app.py. Assicurati che il file esista nella directory del progetto.")
            print("\nPuoi utilizzare i seguenti strumenti:")
            print("- batch_img2img.py: per elaborare batch di immagini")
            print("- extract_frames.py: per estrarre frame da video")
            print("- create_video.py: per creare video da immagini")
            print("- convert_to_mp4.py: per convertire video in formato MP4")
            
            print("\nEsempio di utilizzo:")
            print("python batch_img2img.py --input_dir ./input --output_dir ./output --prompt \"il tuo prompt qui\" --model_path ./checkpoints/sd_xl_turbo_1.0_fp16.safetensors --strength 0.75 --steps 20")
    else:
        print("\nL'applicazione non può essere avviata perché alcune dipendenze non sono soddisfatte.")
        print("Assicurati di installare FFmpeg e tutte le dipendenze richieste prima di avviare l'applicazione.")

if __name__ == "__main__":
    main()