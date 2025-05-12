import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import os
import threading
import sys
import time
from PIL import Image, ImageTk
import glob

# Importa le funzioni dal nuovo script principale
from video_img2img import process_video

class RedirectText:
    def __init__(self, text_widget):
        self.text_widget = text_widget
        self.buffer = ""

    def write(self, string):
        self.buffer += string
        # Aggiorna il widget di testo nella thread principale
        self.text_widget.after(10, self.update_text_widget)
    
    def update_text_widget(self):
        self.text_widget.configure(state=tk.NORMAL)
        self.text_widget.insert(tk.END, self.buffer)
        self.text_widget.see(tk.END)
        self.text_widget.configure(state=tk.DISABLED)
        self.buffer = ""
    
    def flush(self):
        pass

class VideoProcessorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Video Processor con Stable Diffusion")
        self.root.geometry("800x700")
        
        # Imposta l'icona dell'applicazione
        try:
            icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "skull.png")
            if os.path.exists(icon_path):
                icon_image = Image.open(icon_path)
                icon_photo = ImageTk.PhotoImage(icon_image)
                self.root.iconphoto(True, icon_photo)
                print(f"Icona caricata da: {icon_path}")
            else:
                print(f"File icona non trovato: {icon_path}")
        except Exception as e:
            print(f"Errore nel caricamento dell'icona: {str(e)}")
        
        # Variabili
        self.video_path = tk.StringVar()
        self.output_dir = tk.StringVar(value=os.path.join(os.getcwd(), "output"))
        self.prompt = tk.StringVar()
        self.model_path = tk.StringVar()
        self.strength = tk.DoubleVar(value=0.75)
        self.steps = tk.IntVar(value=20)
        self.max_size_mb = tk.DoubleVar(value=16)
        self.quality = tk.IntVar(value=70)
        self.processing = False
        self.no_size_limit = tk.BooleanVar(value=False)  # Nuova variabile per l'opzione senza limiti
        
        # Variabile per il timer
        self.timer_running = False
        self.start_time = 0
        self.timer_text = tk.StringVar(value="Tempo: 00:00:00")
        
        # Variabili per la modalità test
        self.test_mode = tk.BooleanVar(value=False)
        self.test_frames = tk.IntVar(value=10)
        
        # Nuove variabili per upscaling
        self.upscale_height = tk.IntVar(value=1080)
        self.use_realesrgan = tk.BooleanVar(value=True)
        
        # Carica automaticamente il primo modello disponibile
        self.load_default_model()
        
        # Configurazione dell'interfaccia
        self.setup_ui()
        
        # Reindirizza stdout al widget di testo
        self.redirect_stdout()
    
    def load_default_model(self):
        """Carica automaticamente il primo modello disponibile nella cartella models"""
        models_dir = os.path.join(os.getcwd(), "checkpoints")
        
        # Verifica se la cartella models esiste
        if not os.path.exists(models_dir):
            print(f"Cartella checkpoints non trovata in {os.getcwd()}")
            return
        
        # Cerca tutti i file modello nella cartella models
        model_extensions = ["*.safetensors", "*.ckpt", "*.pth"]
        model_files = []
        
        for ext in model_extensions:
            model_files.extend(glob.glob(os.path.join(models_dir, ext)))
        
        # Se ci sono modelli, imposta il primo come predefinito
        if model_files:
            self.model_path.set(model_files[0])
            print(f"Modello caricato automaticamente: {os.path.basename(model_files[0])}")
        else:
            print("Nessun modello trovato nella cartella checkpoints")
    
    def redirect_stdout(self):
        self.stdout_backup = sys.stdout
        sys.stdout = RedirectText(self.log_text)
    
    def restore_stdout(self):
        sys.stdout = self.stdout_backup
    
    def setup_ui(self):
        # Frame principale
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Selezione file video
        ttk.Label(main_frame, text="File Video:").grid(row=0, column=0, sticky=tk.W, pady=5)
        ttk.Entry(main_frame, textvariable=self.video_path, width=50).grid(row=0, column=1, pady=5, sticky=tk.EW)
        ttk.Button(main_frame, text="Sfoglia...", command=self.browse_video).grid(row=0, column=2, pady=5)
        
        # Directory di output
        ttk.Label(main_frame, text="Directory Output:").grid(row=1, column=0, sticky=tk.W, pady=5)
        ttk.Entry(main_frame, textvariable=self.output_dir, width=50).grid(row=1, column=1, pady=5, sticky=tk.EW)
        ttk.Button(main_frame, text="Sfoglia...", command=self.browse_output_dir).grid(row=1, column=2, pady=5)
        
        # Prompt
        ttk.Label(main_frame, text="Prompt:").grid(row=2, column=0, sticky=tk.W, pady=5)
        ttk.Entry(main_frame, textvariable=self.prompt, width=50).grid(row=2, column=1, columnspan=2, pady=5, sticky=tk.EW)
        
        # Modello
        ttk.Label(main_frame, text="Modello:").grid(row=3, column=0, sticky=tk.W, pady=5)
        ttk.Entry(main_frame, textvariable=self.model_path, width=50).grid(row=3, column=1, pady=5, sticky=tk.EW)
        ttk.Button(main_frame, text="Sfoglia...", command=self.browse_model).grid(row=3, column=2, pady=5)
        
        # Parametri
        params_frame = ttk.LabelFrame(main_frame, text="Parametri", padding="10")
        params_frame.grid(row=4, column=0, columnspan=3, sticky=tk.EW, pady=10)
        
        # Strength
        ttk.Label(params_frame, text="Strength:").grid(row=0, column=0, sticky=tk.W, pady=5)
        strength_scale = ttk.Scale(params_frame, variable=self.strength, from_=0.1, to=1.0, orient=tk.HORIZONTAL)
        strength_scale.grid(row=0, column=1, sticky=tk.EW, pady=5)
        strength_label = ttk.Label(params_frame, width=5)
        strength_label.grid(row=0, column=2, pady=5)
        
        # Aggiorna il valore visualizzato quando la scala cambia
        def update_strength_label(*args):
            strength_label.config(text=f"{self.strength.get():.2f}")
        self.strength.trace_add("write", update_strength_label)
        update_strength_label()  # Inizializza il valore
        
        # Steps
        ttk.Label(params_frame, text="Steps:").grid(row=1, column=0, sticky=tk.W, pady=5)
        steps_scale = ttk.Scale(params_frame, variable=self.steps, from_=1, to=50, orient=tk.HORIZONTAL)
        steps_scale.grid(row=1, column=1, sticky=tk.EW, pady=5)
        steps_label = ttk.Label(params_frame, width=5)
        steps_label.grid(row=1, column=2, pady=5)
        
        def update_steps_label(*args):
            steps_label.config(text=str(self.steps.get()))
        self.steps.trace_add("write", update_steps_label)
        update_steps_label()  # Inizializza il valore
        
        # Upscale Height
        ttk.Label(params_frame, text="Altezza Upscale:").grid(row=2, column=0, sticky=tk.W, pady=5)
        upscale_height_values = [720, 1080, 1440, 2160]
        upscale_height_combo = ttk.Combobox(params_frame, textvariable=self.upscale_height, values=upscale_height_values, width=5)
        upscale_height_combo.grid(row=2, column=1, sticky=tk.W, pady=5)
        
        # Checkbox per Real-ESRGAN
        realesrgan_check = ttk.Checkbutton(params_frame, text="Usa Real-ESRGAN per upscaling", variable=self.use_realesrgan)
        realesrgan_check.grid(row=2, column=2, sticky=tk.W, pady=5)
        
        # Max Size (MB)
        max_size_frame = ttk.Frame(params_frame)
        max_size_frame.grid(row=3, column=0, columnspan=3, sticky=tk.EW, pady=5)
        
        ttk.Label(max_size_frame, text="Max Size (MB):").pack(side=tk.LEFT, padx=(0, 5))
        
        # Checkbox per nessun limite di dimensione
        no_limit_check = ttk.Checkbutton(max_size_frame, text="Nessun limite", variable=self.no_size_limit)
        no_limit_check.pack(side=tk.RIGHT)
        
        # Frame per slider e label
        size_slider_frame = ttk.Frame(params_frame)
        size_slider_frame.grid(row=4, column=0, columnspan=3, sticky=tk.EW)
        
        max_size_scale = ttk.Scale(size_slider_frame, variable=self.max_size_mb, from_=1, to=100, orient=tk.HORIZONTAL)
        max_size_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        
        max_size_label = ttk.Label(size_slider_frame, width=5)
        max_size_label.pack(side=tk.RIGHT)
        
        def update_max_size_label(*args):
            max_size_label.config(text=str(int(self.max_size_mb.get())))
        
        def update_max_size_state(*args):
            state = "disabled" if self.no_size_limit.get() else "normal"
            max_size_scale.configure(state=state)  # Usa configure invece di state
        
        self.max_size_mb.trace_add("write", update_max_size_label)
        self.no_size_limit.trace_add("write", update_max_size_state)
        update_max_size_label()  # Inizializza il valore
        update_max_size_state()  # Inizializza lo stato
        
        # Quality
        ttk.Label(params_frame, text="Qualità (0-100):").grid(row=5, column=0, sticky=tk.W, pady=5)
        quality_scale = ttk.Scale(params_frame, variable=self.quality, from_=0, to=100, orient=tk.HORIZONTAL)
        quality_scale.grid(row=5, column=1, sticky=tk.EW, pady=5)
        quality_label = ttk.Label(params_frame, width=5)
        quality_label.grid(row=5, column=2, pady=5)
        
        def update_quality_label(*args):
            quality_label.config(text=str(self.quality.get()))
        self.quality.trace_add("write", update_quality_label)
        update_quality_label()  # Inizializza il valore
        
        # Aggiungi modalità test
        test_frame = ttk.LabelFrame(main_frame, text="Modalità Test", padding="10")
        test_frame.grid(row=5, column=0, columnspan=3, sticky=tk.EW, pady=10)
        
        # Checkbox per attivare la modalità test
        test_check = ttk.Checkbutton(test_frame, text="Attiva modalità test", variable=self.test_mode)
        test_check.grid(row=0, column=0, sticky=tk.W, pady=5)
        
        # Numero di frames da elaborare in modalità test
        ttk.Label(test_frame, text="Numero di frames:").grid(row=0, column=1, sticky=tk.W, pady=5, padx=(20, 0))
        test_frames_entry = ttk.Spinbox(test_frame, from_=1, to=100, textvariable=self.test_frames, width=5)
        test_frames_entry.grid(row=0, column=2, sticky=tk.W, pady=5)
        
        # Funzione per abilitare/disabilitare il campo numero frames
        def update_test_frames_state(*args):
            state = "normal" if self.test_mode.get() else "disabled"
            test_frames_entry.config(state=state)
        
        self.test_mode.trace_add("write", update_test_frames_state)
        update_test_frames_state()  # Inizializza lo stato
        
        # Frame per i pulsanti e il timer
        buttons_frame = ttk.Frame(main_frame)
        buttons_frame.grid(row=6, column=0, columnspan=3, pady=10, sticky=tk.EW)
        
        # Timer display
        timer_label = ttk.Label(buttons_frame, textvariable=self.timer_text)
        timer_label.pack(side=tk.LEFT, padx=(0, 20))
        
        # Pulsante di elaborazione
        self.process_button = ttk.Button(buttons_frame, text="Elabora Video", command=self.process_video_thread)
        self.process_button.pack(side=tk.LEFT, padx=5)
        
        # Pulsante per aprire la cartella di output
        open_folder_button = ttk.Button(buttons_frame, text="Apri Cartella Output", command=self.open_output_folder)
        open_folder_button.pack(side=tk.LEFT, padx=5)
        
        # Pulsante di chiusura
        close_button = ttk.Button(buttons_frame, text="Chiudi", command=self.close_application)
        close_button.pack(side=tk.RIGHT, padx=5)
        
        # Barra di progresso
        self.progress = ttk.Progressbar(main_frame, orient=tk.HORIZONTAL, length=100, mode='indeterminate')
        self.progress.grid(row=7, column=0, columnspan=3, sticky=tk.EW, pady=5)
        
        # Area log
        log_frame = ttk.LabelFrame(main_frame, text="Log", padding="10")
        log_frame.grid(row=8, column=0, columnspan=3, sticky=tk.NSEW, pady=5)
        
        self.log_text = tk.Text(log_frame, height=15, width=70, wrap=tk.WORD)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Aggiungi scrollbar
        scrollbar = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.configure(yscrollcommand=scrollbar.set, state=tk.DISABLED)
        
        # Configura il ridimensionamento
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(8, weight=1)
        
        # Configura il ridimensionamento dei parametri
        for i in range(3):
            params_frame.columnconfigure(1, weight=1)
    
    def browse_video(self):
        filename = filedialog.askopenfilename(
            title="Seleziona Video",
            filetypes=(("File Video", "*.mp4 *.avi *.mov *.mkv"), ("Tutti i file", "*.*"))
        )
        if filename:
            self.video_path.set(filename)
    
    def browse_output_dir(self):
        directory = filedialog.askdirectory(title="Seleziona Directory di Output")
        if directory:
            self.output_dir.set(directory)
    
    def browse_model(self):
        filename = filedialog.askopenfilename(
            title="Seleziona Modello",
            filetypes=(("File Modello", "*.safetensors *.ckpt *.pth"), ("Tutti i file", "*.*"))
        )
        if filename:
            self.model_path.set(filename)
    
    def open_output_folder(self):
        """Apre la cartella di output nel File Explorer"""
        output_dir = self.output_dir.get()
        if os.path.exists(output_dir):
            # Utilizzo di 'explorer' per Windows
            os.startfile(output_dir)
        else:
            messagebox.showinfo("Informazione", "La cartella di output non esiste ancora.")
    
    def update_timer(self):
        """Aggiorna il timer se è in esecuzione"""
        if self.timer_running:
            elapsed_time = time.time() - self.start_time
            hours = int(elapsed_time // 3600)
            minutes = int((elapsed_time % 3600) // 60)
            seconds = int(elapsed_time % 60)
            
            self.timer_text.set(f"Tempo: {hours:02d}:{minutes:02d}:{seconds:02d}")
            self.root.after(1000, self.update_timer)
    
    def start_timer(self):
        """Avvia il timer"""
        self.start_time = time.time()
        self.timer_running = True
        self.update_timer()
    
    def stop_timer(self):
        """Ferma il timer"""
        self.timer_running = False
    
    def close_application(self):
        """Chiude l'applicazione"""
        if self.processing:
            if messagebox.askyesno("Conferma", "Un'elaborazione è in corso. Sei sicuro di voler chiudere?"):
                self.restore_stdout()
                self.root.destroy()
        else:
            self.restore_stdout()
            self.root.destroy()
    
    def process_video_thread(self):
        # Verifica input
        if not self.video_path.get():
            messagebox.showerror("Errore", "Seleziona un file video")
            return
        
        if not self.prompt.get():
            messagebox.showerror("Errore", "Inserisci un prompt")
            return
        
        if not self.model_path.get():
            messagebox.showerror("Errore", "Seleziona un modello")
            return
        
        if self.processing:
            messagebox.showinfo("Informazione", "Elaborazione già in corso")
            return
        
        # Avvia elaborazione in un thread separato
        self.processing = True
        self.process_button.config(state=tk.DISABLED)
        self.progress.start()
        
        # Avvia il timer
        self.start_timer()
        
        # Pulisci il log
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.delete(1.0, tk.END)
        self.log_text.configure(state=tk.DISABLED)
        
        threading.Thread(target=self.run_processing, daemon=True).start()
    
    def run_processing(self):
        try:
            print(f"Avvio elaborazione video: {self.video_path.get()}")
            
            # Verifica se è attiva la modalità test
            max_frames = None
            if self.test_mode.get():
                print(f"Modalità test attiva: elaborazione di {self.test_frames.get()} frames")
                max_frames = self.test_frames.get()
            
            # Crea il percorso di output per il video
            os.makedirs(self.output_dir.get(), exist_ok=True)
            output_video = os.path.join(
                self.output_dir.get(), 
                f"processed_{os.path.basename(self.video_path.get())}"
            )
            
            # Chiama la funzione di elaborazione dal nuovo script
            process_video(
                input_video=self.video_path.get(),
                output_video=output_video,
                prompt=self.prompt.get(),
                model_path=self.model_path.get(),
                strength=self.strength.get(),
                steps=self.steps.get(),
                target_height=self.upscale_height.get(),
                use_realesrgan=self.use_realesrgan.get(),
                max_frames=max_frames,
                max_size_mb=None if self.no_size_limit.get() else self.max_size_mb.get(),
                quality=self.quality.get()
            )
            
            # Mostra messaggio di completamento
            self.root.after(0, lambda: messagebox.showinfo("Completato", "Elaborazione completata con successo!"))
            print("Elaborazione completata con successo!")
            
        except Exception as e:
            print(f"Errore durante l'elaborazione: {str(e)}")
            self.root.after(0, lambda: messagebox.showerror("Errore", f"Errore durante l'elaborazione: {str(e)}"))
        finally:
            # Ferma il timer
            self.stop_timer()
            
            # Ferma la barra di progresso e riabilita il pulsante
            self.root.after(0, self.progress.stop)
            self.root.after(0, lambda: self.process_button.config(state=tk.NORMAL))
            self.processing = False

if __name__ == "__main__":
    root = tk.Tk()
    app = VideoProcessorApp(root)
    root.mainloop()