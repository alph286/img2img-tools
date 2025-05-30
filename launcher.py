import os
import sys
import subprocess
import platform

def bootstrap_dependencies():
    """Install basic dependencies needed for the launcher to run."""
    print("Installing basic dependencies for launcher...")
    try:
        # Install requests, tqdm and setuptools (which provides pkg_resources)
        subprocess.check_call([sys.executable, "-m", "pip", "install", "requests", "tqdm", "setuptools"])
        print("Basic dependencies installed successfully!")
        
        # Now we can safely import these modules
        global requests, tqdm, pkg_resources, shutil
        import requests
        from tqdm import tqdm
        import pkg_resources
        import shutil
        
        return True
    except subprocess.SubprocessError as e:
        print(f"Error installing basic dependencies: {e}")
        print("Please manually install the required packages with:")
        print("pip install requests tqdm setuptools")
        return False

# Try to import required modules, install them if not available
try:
    import requests
    from tqdm import tqdm
    import pkg_resources
    import shutil
except ImportError:
    if not bootstrap_dependencies():
        sys.exit(1)

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
    # Dipendenze base
    required_packages = {
        # "torch": "torch>=2.0.0 --index-url https://download.pytorch.org/whl/cu118", # Gestito separatamente con versioni pinnate
        "torchvision": "torchvision==0.15.2", # Assicuriamoci che torchvision sia pinnato anche qui sebbene installato prima
        "diffusers": "diffusers>=0.19.0",
        "opencv-python": "opencv-python>=4.5.0",
        "transformers": "transformers>=4.30.0",
        "accelerate": "accelerate>=0.20.0",
        "safetensors": "safetensors>=0.3.0",
        "tqdm": "tqdm>=4.65.0",
        "requests": "requests>=2.28.0",
        "pillow": "pillow>=9.0.0",
        "ffmpeg-python": "ffmpeg-python>=0.2.0",
        "numpy": "numpy<2.0", # Aggiunto per garantire la compatibilità
    }
    
    # Installa tutte le dipendenze base senza verificare se sono già installate
    print("Installazione di tutte le dipendenze base...")
    try:
        # Installa PyTorch con supporto CUDA separatamente
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-cache-dir", "--retries", "5", "--timeout", "60", "torch>=2.1.0", "torchvision>=0.16.0", "torchaudio>=2.1.0", "--index-url", "https://download.pytorch.org/whl/cu118"])
        print("PyTorch con supporto CUDA installato con successo!")
        
        # Installa le altre dipendenze
        # Filtra i pacchetti, escludendo torch e torchvision che sono già stati installati con versioni specifiche.
        other_package_names = [pkg for pkg in required_packages.keys() if pkg not in ["torch", "torchvision"]]
        other_packages_to_install = [required_packages[pkg] for pkg in other_package_names if required_packages[pkg]] # Prende la stringa di requisito
        
        if other_packages_to_install:
            # Usiamo --no-cache-dir per evitare problemi di cache, ma rimuoviamo --upgrade per non alterare torchvision pinnato.
            # Pip installerà/aggiornerà questi pacchetti solo se necessario per soddisfare le versioni minime specificate,
            # cercando di mantenere torchvision==0.15.2.
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-cache-dir"] + other_packages_to_install) # MODIFICATO: Rimosso --upgrade
        print("Tutte le dipendenze base sono state installate con successo!")
    except subprocess.SubprocessError as e:
        print(f"Errore durante l'installazione delle dipendenze: {e}")
        print("Prova a installare manualmente le dipendenze mancanti.")
        return False
    
    # Installa le dipendenze per Real-ESRGAN in ordine specifico
    print("\nInstallazione delle dipendenze per l'upscaling di alta qualità...")
    try:
        # Installa prima basicsr (da GitHub per compatibilità con torchvision più recenti)
        # Manteniamo --force-reinstall per basicsr da GitHub per assicurare la versione corretta
        print("Installazione di basicsr da GitHub...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-cache-dir", "git+https://github.com/XPixelGroup/BasicSR.git#egg=basicsr"])
        
        # Poi installa facexlib
        # Rimuoviamo --force-reinstall per facexlib, pip lo gestirà se necessario
        print("Installazione di facexlib...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-cache-dir", "facexlib>=0.2.5"])
        
        # Infine installa realesrgan (da PyPI)
        # Rimuoviamo --force-reinstall per realesrgan da PyPI
        print("Installazione di realesrgan da PyPI...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-cache-dir", "realesrgan>=0.3.0"])
        
        # Verifica se realesrgan è stato installato correttamente
        try:
            import realesrgan
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer
            print("Real-ESRGAN installato con successo da PyPI!")
        except ImportError as e_import_pypi:
            print(f"Real-ESRGAN (da PyPI) non è stato installato/importato correttamente: {e_import_pypi}")
            print("Tentativo di installazione alternativa di Real-ESRGAN da GitHub...")
            try:
                # Disinstalla di nuovo realesrgan prima di tentare da GitHub per pulizia
                print("Tentativo di disinstallazione di realesrgan prima dell'installazione da GitHub...")
                subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "realesrgan"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
                
                # Prova a installare Real-ESRGAN da GitHub
                print("Installazione di Real-ESRGAN da GitHub...")
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", "--no-cache-dir",
                    "git+https://github.com/xinntao/Real-ESRGAN.git"
                ]) # MODIFICATO: Rimosso --upgrade, Rimosso --force-reinstall
                # Riprova gli import dopo l'installazione da GitHub
                import realesrgan 
                from basicsr.archs.rrdbnet_arch import RRDBNet 
                from realesrgan import RealESRGANer
                print("Real-ESRGAN installato con successo tramite GitHub!")
            except (subprocess.SubprocessError, ImportError) as e_gh_install_import:


                print(f"Installazione/Import di Real-ESRGAN da GitHub fallita: {e_gh_install_import}")
                print("L'applicazione non può procedere. Impossibile installare Real-ESRGAN.")
                return False
    except subprocess.SubprocessError as e:
        print(f"Errore durante l'installazione di una dipendenza per Real-ESRGAN (basicsr, facexlib, o tentativo iniziale di realesrgan): {e}")
        print("L'applicazione non può procedere. Impossibile installare le dipendenze per Real-ESRGAN.")
        return False
    
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

def download_upscaler_model():
    """Scarica il modello Real-ESRGAN per l'upscaling, chiedendo conferma all'utente."""
    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    
    model_filename = "RealESRGAN_x2plus.pth"
    model_path = os.path.join(models_dir, model_filename)
    
    if not os.path.exists(model_path):
        # Aggiunta della richiesta di conferma all'utente
        download_choice = input(f"\nIl modello di upscaling Real-ESRGAN ({model_filename}) non è presente nella cartella '{models_dir}'.\nVuoi scaricarlo ora? (circa 65MB) (sì/no): ").lower()
        if download_choice in ["sì", "si", "s", "yes", "y"]:
            print(f"\nScaricamento del modello {model_filename} in corso...")
            model_url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth"
            try:
                download_model(model_url, model_path) # download_model è una funzione helper esistente
                if os.path.exists(model_path):
                    print(f"\nModello di upscaling scaricato con successo in: {model_path}")
                else:
                    # Questa condizione potrebbe verificarsi se download_model non crea il file nonostante non ci siano eccezioni
                    print(f"\nDownload del modello di upscaling fallito. Il file non è stato trovato in: {model_path} dopo il tentativo di download.")
            except Exception as e:
                print(f"\nErrore durante il download del modello di upscaling: {e}")
                # Se il file esiste (potrebbe essere parziale o corrotto), tentare di rimuoverlo
                if os.path.exists(model_path):
                    try:
                        os.remove(model_path)
                        print(f"Rimosso file potenzialmente corrotto/parziale: {model_path}")
                    except OSError as oe:
                        print(f"Errore durante la rimozione del file {model_path}: {oe}")
        else:
            print(f"\nScaricamento del modello {model_filename} saltato. L'upscaling con Real-ESRGAN potrebbe non funzionare.")
    else:
        print(f"\nIl modello di upscaling {model_filename} è già presente in: {model_path}")
    
    return model_path

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
    
    # Verifica se è installato Real-ESRGAN e scarica il modello se necessario
    try:
        import realesrgan
        print("\nReal-ESRGAN è installato. Verifico il modello di upscaling...")
        upscaler_model_path = download_upscaler_model()
    except ImportError:
        print("\nReal-ESRGAN non è installato. L'upscaling utilizzerà metodi standard.")
    
    # Avvia l'applicazione principale solo se le dipendenze sono soddisfatte e FFmpeg è installato
    if ffmpeg_installed:
        app_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app.py")
        if os.path.exists(app_path):
            print("\nAvvio dell'applicazione img2img-tools...")
            try:
                # Passa eventuali argomenti della riga di comando all'app
                cmd = [sys.executable, app_path] + sys.argv[1:]
                subprocess.run(cmd, check=True)
            except subprocess.SubprocessError:
                print("\nErrore durante l'avvio dell'applicazione.")
        else:
            print("\nImpossibile trovare app.py. Assicurati che il file esista nella directory del progetto.")
            print("\nPuoi utilizzare i seguenti strumenti:")
            print("- batch_img2img.py: per elaborare batch di immagini")
            print("- video_img2img.py: per elaborare video con img2img e upscaling")
            print("- extract_frames.py: per estrarre frame da video")
            print("- create_video.py: per creare video da immagini")
            print("- convert_to_mp4.py: per convertire video in formato MP4")
            
            print("\nEsempi di utilizzo:")
            print("python batch_img2img.py --input_dir ./input --output_dir ./output --prompt \"il tuo prompt qui\" --model_path ./checkpoints/sd_xl_turbo_1.0_fp16.safetensors --strength 0.75 --steps 20")
            print("\npython video_img2img.py --input_video ./input.mp4 --output_video ./output.mp4 --prompt \"il tuo prompt qui\" --model_path ./checkpoints/sd_xl_turbo_1.0_fp16.safetensors --strength 0.75 --steps 1")
    else:
        print("\nL'applicazione non può essere avviata perché alcune dipendenze non sono soddisfatte o l'installazione è fallita.")
        print("Controlla i messaggi di errore precedenti per i dettagli.")
        print("Assicurati di installare FFmpeg e tutte le dipendenze richieste prima di avviare l'applicazione.")
        sys.exit(1)

if __name__ == "__main__":
    main()