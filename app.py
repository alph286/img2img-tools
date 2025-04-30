import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import os
import threading
import sys
from PIL import Image, ImageTk

# Importa le funzioni dal tuo script esistente
from process_video import process_video

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
        
        # Variabili per la modalità test
        self.test_mode = tk.BooleanVar(value=False)
        self.test_frames = tk.IntVar(value=10)
        
        # Configurazione dell'interfaccia
        self.setup_ui()
        
        # Reindirizza stdout al widget di testo
        self.redirect_stdout()
    
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
        
        # Max Size (MB)
        ttk.Label(params_frame, text="Max Size (MB):").grid(row=2, column=0, sticky=tk.W, pady=5)
        max_size_scale = ttk.Scale(params_frame, variable=self.max_size_mb, from_=1, to=100, orient=tk.HORIZONTAL)
        max_size_scale.grid(row=2, column=1, sticky=tk.EW, pady=5)
        max_size_label = ttk.Label(params_frame, width=5)
        max_size_label.grid(row=2, column=2, pady=5)
        
        def update_max_size_label(*args):
            max_size_label.config(text=str(int(self.max_size_mb.get())))
        self.max_size_mb.trace_add("write", update_max_size_label)
        update_max_size_label()  # Inizializza il valore
        
        # Quality
        ttk.Label(params_frame, text="Qualità (0-100):").grid(row=3, column=0, sticky=tk.W, pady=5)
        quality_scale = ttk.Scale(params_frame, variable=self.quality, from_=0, to=100, orient=tk.HORIZONTAL)
        quality_scale.grid(row=3, column=1, sticky=tk.EW, pady=5)
        quality_label = ttk.Label(params_frame, width=5)
        quality_label.grid(row=3, column=2, pady=5)
        
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
        
        # Pulsante di elaborazione
        self.process_button = ttk.Button(main_frame, text="Elabora Video", command=self.process_video_thread)
        self.process_button.grid(row=6, column=0, columnspan=3, pady=10)
        
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
        
        # Pulisci il log
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.delete(1.0, tk.END)
        self.log_text.configure(state=tk.DISABLED)
        
        threading.Thread(target=self.run_processing, daemon=True).start()
    
    def run_processing(self):
        try:
            print(f"Avvio elaborazione video: {self.video_path.get()}")
            
            # Verifica se è attiva la modalità test
            if self.test_mode.get():
                print(f"Modalità test attiva: elaborazione di {self.test_frames.get()} frames")
                max_frames = self.test_frames.get()
            else:
                max_frames = None
            
            # Chiama la funzione di elaborazione dal tuo script
            process_video(
                self.video_path.get(),
                self.output_dir.get(),
                self.prompt.get(),
                self.model_path.get(),
                self.strength.get(),
                self.steps.get(),
                self.max_size_mb.get(),
                self.quality.get(),
                max_frames=max_frames  # Passa il parametro max_frames
            )
            
            # Mostra messaggio di completamento
            self.root.after(0, lambda: messagebox.showinfo("Completato", "Elaborazione completata con successo!"))
            print("Elaborazione completata con successo!")
            
        except Exception as e:
            print(f"Errore durante l'elaborazione: {str(e)}")
            self.root.after(0, lambda: messagebox.showerror("Errore", f"Errore durante l'elaborazione: {str(e)}"))
        finally:
            # Ferma la barra di progresso e riabilita il pulsante
            self.root.after(0, self.progress.stop)
            self.root.after(0, lambda: self.process_button.config(state=tk.NORMAL))
            self.processing = False

if __name__ == "__main__":
    root = tk.Tk()
    app = VideoProcessorApp(root)
    root.mainloop()