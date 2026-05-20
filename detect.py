"""
=============================================================
  Hey Orvyn - Real-Time Wake Word Detection
  Listens to your microphone and detects "Hey Orvyn"
=============================================================

HOW TO RUN:
  python detect.py

WHAT IT DOES:
  1. Opens your microphone
  2. Continuously listens in 10ms chunks
  3. Every chunk: converts audio to mel spectrogram
  4. Feeds spectrogram to your ONNX model
  5. If probability > threshold → WAKE WORD DETECTED!
  6. Press Ctrl+C to stop

HOW REAL-TIME DETECTION WORKS:
  Think of it like a sliding window:
  
  Audio stream: [.......................................................]
  Window 1:     [==========]          (1 second)
  Window 2:      [==========]         (shifted 10ms)
  Window 3:       [==========]        (shifted 10ms again)
  
  Every 10ms we take the last 1 second of audio,
  convert it to a mel spectrogram, and run the model.
  If any window scores > 0.5, we detected the wake word.
  
  This is exactly how Alexa, Siri, and Google Assistant work.
"""

import numpy as np
import onnxruntime as ort
import sounddevice as sd
import librosa
import torch
import torchaudio.transforms as T
from pathlib import Path
from collections import deque
import threading
import time
import json

# ── INSTALL SOUNDDEVICE IF NEEDED ─────────────────────────────────────────────
try:
    import sounddevice as sd
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "sounddevice"])
    import sounddevice as sd

# ── CONFIGURATION ─────────────────────────────────────────────────────────────
ONNX_PATH  = Path("export/hey_orvyn.onnx")
CONFIG_PATH = Path("export/model_config.json")

# Audio settings — must match training exactly
SAMPLE_RATE    = 16000   # 16kHz
WINDOW_SIZE    = 16000   # 1 second of audio
HOP_SIZE       = 160     # 10ms hop (how often we run inference)
CHANNELS       = 1       # mono

# Detection settings
THRESHOLD         = 0.5   # above this = wake word detected
COOLDOWN_SECONDS  = 2.0   # wait 2 seconds before detecting again
                           # prevents multiple triggers from one utterance

# Mel spectrogram settings — must match training exactly
N_MELS     = 40
N_FFT      = 400
HOP_LENGTH = 160
F_MIN      = 80
F_MAX      = 8000


# ── LOAD MODEL AND CONFIG ─────────────────────────────────────────────────────
def load_model():
    """
    Loads the ONNX model for inference.
    
    ONNX Runtime automatically selects the best execution provider:
    - CUDA  : uses your RTX 4060 GPU (fastest)
    - CPU   : uses your CPU (still very fast for this tiny model)
    
    For a 94KB model, CPU is actually fine — 0.18ms per inference.
    """
    if not ONNX_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {ONNX_PATH}\n"
            "Run export_model.py first!"
        )
    
    # Try GPU first, fall back to CPU
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    session   = ort.InferenceSession(str(ONNX_PATH), providers=providers)
    
    # Check which provider is being used
    provider = session.get_providers()[0]
    print(f"  Using: {provider}")
    
    return session


def load_config():
    """Loads model configuration from JSON."""
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            return json.load(f)
    return {}


# ── MEL SPECTROGRAM TRANSFORM ─────────────────────────────────────────────────
# Create once and reuse — same settings as training
mel_transform = T.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=N_FFT,
    hop_length=HOP_LENGTH,
    n_mels=N_MELS,
    f_min=F_MIN,
    f_max=F_MAX,
    power=2.0,
)


def audio_to_mel(audio_np):
    """
    Converts 1 second of raw audio to mel spectrogram.
    
    This is the EXACT same pipeline used during training.
    If we use different settings here, the model won't work.
    
    Input:  numpy array of shape (16000,) — 1 second at 16kHz
    Output: numpy array of shape (1, 1, 40, 101) — ready for model
    """
    # Convert to torch tensor
    waveform = torch.tensor(audio_np, dtype=torch.float32).unsqueeze(0)
    
    # Compute mel spectrogram: (1, 40, 101)
    mel_spec = mel_transform(waveform)
    
    # Log scale
    mel_spec = torch.log(mel_spec + 1e-9)
    
    # Normalize
    mean = mel_spec.mean()
    std  = mel_spec.std()
    if std > 0:
        mel_spec = (mel_spec - mean) / std
    
    # Add batch dimension: (1, 40, 101) → (1, 1, 40, 101)
    return mel_spec.unsqueeze(0).numpy()


# ── REAL-TIME DETECTOR CLASS ──────────────────────────────────────────────────
class WakeWordDetector:
    """
    Real-time wake word detector using sliding window approach.
    
    HOW IT WORKS:
    1. Audio stream feeds chunks into a circular buffer
    2. Every HOP_SIZE samples, we take the last WINDOW_SIZE samples
    3. Convert to mel spectrogram
    4. Run ONNX model
    5. Check if probability > threshold
    
    CIRCULAR BUFFER (deque):
    Like a conveyor belt — new audio comes in from one end,
    old audio falls off the other end.
    Always contains exactly the last 1 second of audio.
    
    maxlen=WINDOW_SIZE means it automatically drops old samples.
    """
    
    def __init__(self, session):
        self.session       = session
        self.buffer        = deque(maxlen=WINDOW_SIZE)
        self.last_detected = 0
        self.is_running    = False
        self.hop_counter   = 0
        
        # Pre-fill buffer with silence
        self.buffer.extend([0.0] * WINDOW_SIZE)
        
        # Statistics
        self.total_frames    = 0
        self.total_detections = 0
        self.max_prob        = 0.0
    
    def process_chunk(self, audio_chunk):
        """
        Processes one chunk of audio from the microphone.
        
        Called by sounddevice callback every HOP_SIZE samples (10ms).
        Runs the full detection pipeline in real-time.
        """
        # Add new audio to buffer
        self.buffer.extend(audio_chunk.flatten().tolist())
        self.hop_counter += len(audio_chunk)
        self.total_frames += 1
        
        # Run inference every HOP_SIZE samples
        if self.hop_counter >= HOP_SIZE:
            self.hop_counter = 0
            
            # Convert buffer to numpy array
            audio_np = np.array(self.buffer, dtype=np.float32)
            
            # Convert to mel spectrogram
            mel_input = audio_to_mel(audio_np)
            
            # Run ONNX model
            outputs = self.session.run(
                None,
                {'mel_spectrogram': mel_input}
            )
            
            # Get probability (apply sigmoid to logit)
            logit = outputs[0][0][0]
            prob  = 1.0 / (1.0 + np.exp(-logit))
            
            # Update max probability for display
            self.max_prob = max(self.max_prob, prob)
            
            # Check detection
            current_time = time.time()
            cooldown_ok  = (current_time - self.last_detected) > COOLDOWN_SECONDS
            
            if prob > THRESHOLD and cooldown_ok:
                self.last_detected     = current_time
                self.total_detections += 1
                self.on_wake_word_detected(prob)
            
            # Show live probability bar
            self.display_probability(prob)
    
    def on_wake_word_detected(self, probability):
        """
        Called when wake word is detected.
        
        In your real app, this is where you:
        - Start recording the user's command
        - Call your AI assistant API
        - Show visual feedback on screen
        - Play a sound effect
        
        For now we just print a big notification.
        """
        print("\n" + "🔥" * 30)
        print(f"  🎤  HEY ORVYN DETECTED!")
        print(f"  Confidence: {probability*100:.1f}%")
        print(f"  Detection #{self.total_detections}")
        print("🔥" * 30 + "\n")
    
    def display_probability(self, prob):
        """
        Shows a live probability bar in the terminal.
        Updates in place (no scrolling).
        
        Green bar = safe (below threshold)
        Red bar   = detected (above threshold)
        """
        bar_length = 40
        filled     = int(prob * bar_length)
        bar        = "█" * filled + "░" * (bar_length - filled)
        
        # Color based on probability
        if prob > THRESHOLD:
            status = "🔥 DETECTED!"
            color  = "\033[91m"  # red
        elif prob > 0.3:
            status = "⚡ Close..."
            color  = "\033[93m"  # yellow
        else:
            status = "👂 Listening"
            color  = "\033[92m"  # green
        
        reset = "\033[0m"
        
        # \r moves cursor to start of line (overwrites previous output)
        print(
            f"\r{color}[{bar}]{reset} "
            f"{prob:.3f} {status}    ",
            end='',
            flush=True
        )


# ── MICROPHONE STREAM ─────────────────────────────────────────────────────────
def start_detection(detector):
    """
    Opens microphone and starts real-time detection.
    
    HOW SOUNDDEVICE WORKS:
    sounddevice opens your microphone and calls our callback
    function every time a new chunk of audio is ready.
    
    The callback runs in a separate thread automatically.
    We just define what to do with each chunk — detector.process_chunk().
    
    blocksize=HOP_SIZE means callback is called every 160 samples (10ms).
    dtype='float32' means audio is already in float format (-1.0 to 1.0).
    """
    print("\n🎤 Microphone opened successfully!")
    print(f"   Sample rate : {SAMPLE_RATE} Hz")
    print(f"   Window size : {WINDOW_SIZE/SAMPLE_RATE*1000:.0f} ms")
    print(f"   Hop size    : {HOP_SIZE/SAMPLE_RATE*1000:.0f} ms")
    print(f"   Threshold   : {THRESHOLD}")
    print(f"   Cooldown    : {COOLDOWN_SECONDS}s\n")
    print("Say 'Hey Orvyn' to test your model!")
    print("Press Ctrl+C to stop\n")
    print("─" * 60)
    
    def audio_callback(indata, frames, time_info, status):
        """Called by sounddevice for each audio chunk."""
        if status:
            pass  # ignore status messages
        detector.process_chunk(indata[:, 0])  # use first channel only
    
    # Open microphone stream
    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype='float32',
        blocksize=HOP_SIZE,
        callback=audio_callback
    ):
        try:
            # Keep running until Ctrl+C
            while True:
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("\n\n" + "="*60)
            print("  DETECTION STOPPED")
            print("="*60)
            print(f"  Total frames    : {detector.total_frames}")
            print(f"  Total detections: {detector.total_detections}")
            print(f"  Max probability : {detector.max_prob:.4f}")
            print("="*60)


# ── LIST AVAILABLE MICROPHONES ────────────────────────────────────────────────
def list_microphones():
    """
    Lists all available microphones on your system.
    Helps if the default mic isn't working.
    """
    print("\n📋 Available microphones:")
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        if device['max_input_channels'] > 0:
            default = " ← DEFAULT" if i == sd.default.device[0] else ""
            print(f"  [{i}] {device['name']}{default}")
    print()


# ── ENTRY POINT ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("="*60)
    print("  HEY ORVYN — REAL-TIME DETECTION")
    print("  Listening for your wake word...")
    print("="*60)
    
    # Load model
    print("\n📂 Loading ONNX model...")
    try:
        session = load_model()
        config  = load_config()
        
        if config:
            print(f"  Model    : {config.get('model_name', 'Hey Orvyn')}")
            print(f"  Accuracy : {config['performance']['val_accuracy']*100:.1f}%")
            print(f"  FAR      : {config['performance']['val_far']*100:.2f}%")
            print(f"  FRR      : {config['performance']['val_frr']*100:.2f}%")
        
        print("  ✅ Model loaded!")
        
    except FileNotFoundError as e:
        print(f"  ❌ {e}")
        exit(1)
    
    # List microphones
    list_microphones()
    
    # Create detector
    detector = WakeWordDetector(session)
    
    # Start real-time detection
    start_detection(detector)