# Image to Image Processing Tools

Una collezione di strumenti personali per la sperimentazione con modelli di diffusione per la generazione e manipolazione di immagini.

## Funzionalità

- `batch_img2img.py`: Elabora un batch di immagini utilizzando un modello di diffusione
- `extract_frames.py`: Estrae i frame da un file video
- `create_video.py`: Crea un video da una sequenza di immagini con controllo della dimensione del file
- `convert_to_mp4.py`: Converte video in formato MP4 compatibile
- `app.py`: Interfaccia grafica per l'elaborazione di video

## Requisiti

### Dipendenze Python

- Python 3.8+
- PyTorch con supporto CUDA (per GPU)
- `diffusers` (≥ 0.19.0)
- `opencv-python` (≥ 4.5.0)
- `transformers` (≥ 4.30.0)
- `accelerate` (≥ 0.20.0)
- `safetensors` (≥ 0.3.0)
- `tqdm` (≥ 4.65.0)
- `requests` (≥ 2.28.0)

### Dipendenze Esterne

- FFmpeg (per la compressione video)

## Installazione

### Windows

1. Installa Python 3.8 o superiore da [https://www.python.org/](https://www.python.org/)
   - Assicurati di selezionare **"Aggiungi Python al PATH"** durante l'installazione.
2. Installa FFmpeg:
   - Vai su: [https://www.gyan.dev/ffmpeg/builds/](https://www.gyan.dev/ffmpeg/builds/)
   - Scarica `ffmpeg-release-essentials.zip`
   - Estrai in `C:\ffmpeg`
   - Aggiungi `C:\ffmpeg\bin` al `PATH` di sistema
3. Avvia il launcher:
   - Doppio clic su `launcher.bat` (per utenti meno esperti)
   - Il launcher verificherà e installerà automaticamente tutte le dipendenze Python necessarie

### macOS

1. Installa Python:
   ```bash
   brew install python
   ```

2. Installa FFmpeg:
   ```bash
   brew install ffmpeg
   ```

3. Avvia il launcher:
   ```bash
   python launcher.py
   ```

## Modelli

Per utilizzare `batch_img2img.py`, è necessario scaricare un modello di diffusione compatibile:

- [SDXL Turbo](https://huggingface.co/stabilityai/sdxl-turbo) – Modello consigliato per la velocità
- [Stable Diffusion XL](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) – Per risultati di qualità superiore

Scaricare il modello in formato `.safetensors` e posizionarlo nella directory `checkpoints/` del progetto.

## Utilizzo

### Avvio Rapido (Windows)

Esegui `launcher.bat`. Il launcher:

- Verifica la presenza di Python
- Installa le dipendenze Python
- Verifica FFmpeg
- Offrirà di scaricare il modello SDXL Turbo se non presente
- Avvierà `app.py` (interfaccia grafica)

Su macOS o Linux, usa:

```bash
python launcher.py
```

### Elaborazione batch di immagini

```bash
python batch_img2img.py \
  --input_dir ./input \
  --output_dir ./output \
  --prompt "il tuo prompt qui" \
  --model_path ./checkpoints/sd_xl_turbo_1.0_fp16.safetensors \
  --strength 0.75 \
  --steps 20
```

### Estrazione frame da video

```bash
python extract_frames.py \
  --video ./input.mp4 \
  --output_dir ./frames \
  --extract_all
```

### Creazione di video da immagini

```bash
python create_video.py \
  --input_dir ./frames \
  --output ./output.mp4 \
  --fps 30 \
  --max_size_mb 16
```

## Risoluzione dei problemi

- **Errore "CUDA non disponibile"**  
  Assicurati di avere una GPU NVIDIA compatibile e driver aggiornati. Installa PyTorch con supporto CUDA:
  ```bash
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
  ```

- **Errore "FFmpeg non trovato"**  
  Verifica che FFmpeg sia installato e che `ffmpeg` sia nel `PATH` di sistema.

- **Errori di memoria**  
  Riduci la dimensione delle immagini o usa un modello più leggero come SDXL Turbo.

---

Questo è un progetto personale per sperimentazione e non è destinato all'uso in produzione.  
Completamente in vibe coding.