"""
=============================================================
  Hey Orvyn - Audio Preprocessing Script
  Converts raw .wav files into mel spectrograms for training
=============================================================

HOW TO RUN:
  python preprocess.py

WHAT IT DOES:
  1. Reads every .wav file from data/raw/positive/
  2. Resamples to 16kHz mono (standard for speech ML)
  3. Trims silence from start and end
  4. Pads or crops to exactly 1 second
  5. Converts to 40-band log-mel spectrogram
  6. Normalizes the spectrogram
  7. Saves as .npy file in data/processed/positive/
  8. Does the same for negative samples (if any exist)

WHY PREPROCESSING?
  Raw audio = list of 16,000 numbers per second
  Neural networks can't learn patterns from raw numbers easily.
  Mel spectrogram = 2D image (40 freq bands × 101 time steps)
  Neural networks are excellent at learning patterns from images.
  This conversion is the bridge between raw audio and ML.

OUTPUT:
  data/processed/positive/<language>/<filename>.npy
  data/processed/metadata.csv  (maps each file to its label)
"""

# ── IMPORTS ──────────────────────────────────────────────────────────────────
import os
import numpy as np
import soundfile as sf
import librosa
import torch
import torchaudio
import torchaudio.transforms as T
from pathlib import Path
from tqdm import tqdm
import csv
import warnings
warnings.filterwarnings('ignore')

# ── CONFIGURATION ─────────────────────────────────────────────────────────────
#
# These are the exact settings used by openWakeWord and most
# production wake word systems. Don't change these unless you
# know what you're doing — they must match during training AND
# inference (when the model runs on your phone/PC).
#
# SAMPLE_RATE = 16000
#   Standard for speech ML. Human voice is 80Hz-8kHz.
#   16kHz captures everything important, wastes no space.
#
# N_MELS = 40
#   Number of frequency bands in our spectrogram.
#   40 bands is the sweet spot — enough detail, not too large.
#
# N_FFT = 400
#   Window size for FFT (Fast Fourier Transform).
#   400 samples at 16kHz = 25ms window — standard for speech.
#
# HOP_LENGTH = 160
#   How many samples we move forward each step.
#   160 samples at 16kHz = 10ms hop — standard for speech.
#
# MAX_LENGTH_SECONDS = 1.0
#   All clips padded/cropped to exactly 1 second.
#   "Hey Orvyn" takes about 0.6-0.8 seconds to say.
#   1 second gives enough context before and after.
#
# After processing: each clip becomes a (40, 101) array
#   40  = frequency bands (mel filters)
#   101 = time steps (1 second / 10ms hop = 100, +1 = 101)

SAMPLE_RATE       = 16000
N_MELS            = 40
N_FFT             = 400
HOP_LENGTH        = 160
MAX_LENGTH_SAMPLES = int(SAMPLE_RATE * 1.0)  # exactly 16000 samples = 1 second

# Input folders
POSITIVE_RAW_DIR  = Path("data/raw/positive")
NEGATIVE_RAW_DIR  = Path("data/raw/negative")

# Output folders
POSITIVE_PROC_DIR = Path("data/processed/positive")
NEGATIVE_PROC_DIR = Path("data/processed/negative")
METADATA_FILE     = Path("data/processed/metadata.csv")

# ── CREATE OUTPUT FOLDERS ─────────────────────────────────────────────────────
def create_output_folders():
    """
    Creates all necessary output folders before processing starts.
    Mirrors the language folder structure from raw to processed.
    """
    POSITIVE_PROC_DIR.mkdir(parents=True, exist_ok=True)
    NEGATIVE_PROC_DIR.mkdir(parents=True, exist_ok=True)
    
    # Mirror language subfolders from raw to processed
    if POSITIVE_RAW_DIR.exists():
        for lang_folder in POSITIVE_RAW_DIR.iterdir():
            if lang_folder.is_dir():
                (POSITIVE_PROC_DIR / lang_folder.name).mkdir(
                    parents=True, exist_ok=True
                )
    
    print("✅ Output folders created")


# ── MEL SPECTROGRAM TRANSFORM ─────────────────────────────────────────────────
#
# WHAT IS A MEL SPECTROGRAM?
# 
# Normal spectrogram: shows frequency vs time, using Hz (cycles/second)
# Mel spectrogram: same thing but frequency axis uses "mel scale"
#
# The mel scale mimics how HUMAN EARS hear sound.
# Our ears are more sensitive to differences in low frequencies
# than high frequencies. For example:
#   - We easily hear the difference between 100Hz and 200Hz
#   - We struggle to hear the difference between 8000Hz and 8100Hz
#
# The mel scale compresses high frequencies and expands low ones,
# matching how we actually perceive sound. This makes it much
# easier for neural networks to learn speech patterns.
#
# We create this transform ONCE and reuse it for every file.
# This is efficient — no need to recreate it 1709 times.

mel_transform = T.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=N_FFT,
    hop_length=HOP_LENGTH,
    n_mels=N_MELS,
    f_min=80,    # minimum frequency: 80Hz (below human voice range)
    f_max=8000,  # maximum frequency: 8kHz (above most consonants)
    power=2.0,   # power spectrogram (amplitude squared)
)


# ── AUDIO LOADING AND RESAMPLING ──────────────────────────────────────────────
def load_audio(file_path):
    try:
        import soundfile as sf
        # Load with soundfile instead of torchaudio
        audio_np, sr = sf.read(str(file_path))
        
        # Convert stereo to mono
        if len(audio_np.shape) > 1:
            audio_np = audio_np.mean(axis=1)
        
        # Convert to float32
        audio_np = audio_np.astype(np.float32)
        
        # Convert to torch tensor (1, samples)
        waveform = torch.tensor(audio_np).unsqueeze(0)
        
        # Resample to 16kHz if needed
        if sr != SAMPLE_RATE:
            resampler = T.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)
            waveform = resampler(waveform)
        
        return waveform
        
    except Exception as e:
        return None

# ── SILENCE TRIMMING ──────────────────────────────────────────────────────────
def trim_silence(waveform):
    """
    Removes silence from the beginning and end of audio.
    
    WHY THIS MATTERS:
    TTS APIs sometimes add silence before/after speech.
    Example: "____Hey Orvyn____" (underscores = silence)
    
    If we don't trim, the model learns that silence is part
    of "Hey Orvyn" — which is wrong. Real users won't have
    clean silence before they speak.
    
    top_db=30 means: remove anything quieter than 30dB below
    the loudest point. This removes silence but keeps speech.
    """
    # Convert to numpy for librosa
    audio_np = waveform.squeeze().numpy()
    
    # librosa.effects.trim returns (trimmed_audio, trim_indices)
    trimmed, _ = librosa.effects.trim(audio_np, top_db=30)
    
    # Convert back to torch tensor with channel dimension
    return torch.tensor(trimmed).unsqueeze(0)


# ── PAD OR CROP TO FIXED LENGTH ───────────────────────────────────────────────
def pad_or_crop(waveform):
    """
    Makes every audio clip exactly 1 second (16000 samples).
    
    WHY FIXED LENGTH?
    Neural networks need inputs of the same size.
    Like how a form has fixed-size boxes — you can't submit
    a form with boxes of different sizes.
    
    If audio is SHORT (< 1 sec):
        We pad with zeros (silence) on the RIGHT side.
        "Hey Orvyn" = 0.7s → add 0.3s silence at end
        This is fine — silence doesn't confuse the model.
    
    If audio is LONG (> 1 sec):
        We crop from the CENTER of the audio.
        WHY CENTER? The wake word is most likely in the middle.
        Cropping from start/end might cut off the word.
    """
    num_samples = waveform.shape[1]
    
    if num_samples < MAX_LENGTH_SAMPLES:
        # PAD: add zeros at the end
        padding = MAX_LENGTH_SAMPLES - num_samples
        waveform = torch.nn.functional.pad(waveform, (0, padding))
        
    elif num_samples > MAX_LENGTH_SAMPLES:
        # CROP: take from center
        start = (num_samples - MAX_LENGTH_SAMPLES) // 2
        waveform = waveform[:, start:start + MAX_LENGTH_SAMPLES]
    
    return waveform


# ── CONVERT TO MEL SPECTROGRAM ────────────────────────────────────────────────
def audio_to_mel_spectrogram(waveform):
    """
    The core transformation: audio waveform → mel spectrogram.
    
    STEP BY STEP:
    1. mel_transform(waveform)
       Input:  (1, 16000) — 1 second of audio
       Output: (1, 40, 101) — mel spectrogram
       
    2. .squeeze(0)
       Remove the channel dimension: (1, 40, 101) → (40, 101)
       
    3. torch.log(spectrogram + 1e-9)
       Convert to log scale (decibels essentially)
       WHY LOG? Human hearing is logarithmic. Also, log scale
       compresses the huge range of energy values (0.0001 to 100)
       into a smaller range (-20 to 4) that's easier to learn from.
       The 1e-9 prevents log(0) = infinity.
       
    4. Normalize: subtract mean, divide by std
       WHY? Makes all spectrograms have similar value ranges.
       Like normalizing exam scores to a 0-100 scale regardless
       of how hard the exam was. Helps training converge faster.
    
    FINAL OUTPUT: numpy array of shape (40, 101)
        40  rows = mel frequency bands (low to high)
        101 cols = time steps (each = 10ms, total = 1.01 seconds)
    """
    # Step 1: compute mel spectrogram
    mel_spec = mel_transform(waveform)  # (1, 40, 101)
    
    # Step 2: remove channel dimension
    mel_spec = mel_spec.squeeze(0)      # (40, 101)
    
    # Step 3: convert to log scale
    mel_spec = torch.log(mel_spec + 1e-9)
    
    # Step 4: normalize (zero mean, unit variance)
    mean = mel_spec.mean()
    std  = mel_spec.std()
    if std > 0:
        mel_spec = (mel_spec - mean) / std
    else:
        mel_spec = mel_spec - mean
    
    # Convert to numpy for saving
    return mel_spec.numpy()


# ── PROCESS ONE FILE ──────────────────────────────────────────────────────────
def process_file(input_path, output_path):
    """
    Full pipeline for one audio file:
    load → trim → pad/crop → mel spectrogram → save
    
    RETURNS: True if successful, False if failed
    """
    # Step 1: Load audio
    waveform = load_audio(input_path)
    if waveform is None:
        return False
    
    # Step 2: Trim silence
    waveform = trim_silence(waveform)
    
    # Safety check — if after trimming audio is empty
    if waveform.shape[1] < 100:
        return False
    
    # Step 3: Pad or crop to exactly 1 second
    waveform = pad_or_crop(waveform)
    
    # Step 4: Convert to mel spectrogram
    mel_spec = audio_to_mel_spectrogram(waveform)
    
    # Step 5: Save as numpy array
    # WHY .npy? Fast to load during training.
    # Loading .npy is 10x faster than loading .wav + processing.
    # During training we load thousands of files — speed matters.
    np.save(str(output_path), mel_spec)
    
    return True


# ── PROCESS ALL POSITIVE SAMPLES ─────────────────────────────────────────────
def process_positive_samples():
    """
    Processes all 1,709 positive "Hey Orvyn" samples.
    Returns list of (npy_path, label, language) for metadata.
    """
    print("\n" + "="*60)
    print("  PROCESSING POSITIVE SAMPLES (Hey Orvyn)")
    print("="*60)
    
    # Collect all wav files
    all_wav_files = list(POSITIVE_RAW_DIR.rglob("*.wav"))
    print(f"  Found {len(all_wav_files)} wav files")
    
    metadata = []
    success  = 0
    failed   = 0
    
    with tqdm(total=len(all_wav_files), desc="Processing", unit="file") as pbar:
        for wav_path in all_wav_files:
            
            # Get language from parent folder name
            # e.g. data/raw/positive/te/te_shubh_0_123.wav → "te"
            language = wav_path.parent.name
            
            # Build output path: same structure but in processed/
            # .wav → .npy
            rel_path   = wav_path.relative_to(POSITIVE_RAW_DIR)
            npy_path   = POSITIVE_PROC_DIR / rel_path.with_suffix('.npy')
            
            # Make sure parent folder exists
            npy_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Skip if already processed
            if npy_path.exists():
                metadata.append((str(npy_path), 1, language))
                success += 1
                pbar.update(1)
                continue
            
            # Process the file
            ok = process_file(wav_path, npy_path)
            
            if ok:
                # label=1 means POSITIVE (this IS "Hey Orvyn")
                metadata.append((str(npy_path), 1, language))
                success += 1
            else:
                failed += 1
            
            pbar.update(1)
            pbar.set_postfix({"✅": success, "❌": failed})
    
    print(f"\n  Positive done: {success} success, {failed} failed")
    return metadata


# ── PROCESS NEGATIVE SAMPLES ──────────────────────────────────────────────────
def process_negative_samples():
    """
    Processes negative samples (non-wake-word speech).
    
    NOTE: We haven't downloaded negatives yet.
    This function handles them when they exist.
    If the folder is empty, it just skips gracefully.
    """
    if not NEGATIVE_RAW_DIR.exists():
        print("\n  ℹ️  No negative samples found yet — skipping")
        print("  We'll add negatives before training.")
        return []
    
    all_wav_files = list(NEGATIVE_RAW_DIR.rglob("*.wav"))
    
    if len(all_wav_files) == 0:
        print("\n  ℹ️  Negative folder empty — skipping")
        return []
    
    print("\n" + "="*60)
    print("  PROCESSING NEGATIVE SAMPLES (not Hey Orvyn)")
    print("="*60)
    print(f"  Found {len(all_wav_files)} negative wav files")
    
    metadata = []
    success  = 0
    failed   = 0
    
    with tqdm(total=len(all_wav_files), desc="Negatives", unit="file") as pbar:
        for wav_path in all_wav_files:
            
            rel_path = wav_path.relative_to(NEGATIVE_RAW_DIR)
            npy_path = NEGATIVE_PROC_DIR / rel_path.with_suffix('.npy')
            npy_path.parent.mkdir(parents=True, exist_ok=True)
            
            if npy_path.exists():
                metadata.append((str(npy_path), 0, "negative"))
                success += 1
                pbar.update(1)
                continue
            
            ok = process_file(wav_path, npy_path)
            
            if ok:
                # label=0 means NEGATIVE (this is NOT "Hey Orvyn")
                metadata.append((str(npy_path), 0, "negative"))
                success += 1
            else:
                failed += 1
            
            pbar.update(1)
    
    print(f"\n  Negatives done: {success} success, {failed} failed")
    return metadata


# ── SAVE METADATA CSV ─────────────────────────────────────────────────────────
def save_metadata(all_metadata):
    """
    Saves a CSV file mapping every processed file to its label.
    
    WHAT IS METADATA?
    A simple table that maps:
    - file path → where the .npy file is
    - label     → 1 (wake word) or 0 (not wake word)
    - language  → which language this sample is in
    
    During training, we read this CSV to know:
    "Load this file and treat it as a positive/negative example"
    
    FORMAT:
    filepath,label,language
    data/processed/positive/en/en_Rachel_0_123.npy,1,en
    data/processed/positive/hi/hi_shubh_0_456.npy,1,hi
    ...
    """
    METADATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    with open(METADATA_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['filepath', 'label', 'language'])
        writer.writerows(all_metadata)
    
    print(f"\n✅ Metadata saved to {METADATA_FILE}")
    print(f"   Total entries: {len(all_metadata)}")


# ── PRINT STATISTICS ──────────────────────────────────────────────────────────
def print_statistics(metadata):
    """
    Shows a breakdown of processed data per language.
    Helps identify if any language is underrepresented.
    """
    print("\n" + "="*60)
    print("  PROCESSING COMPLETE — DATASET STATISTICS")
    print("="*60)
    
    # Count per language
    lang_counts = {}
    for _, label, lang in metadata:
        if label == 1:  # only positives
            lang_counts[lang] = lang_counts.get(lang, 0) + 1
    
    total_positive = sum(1 for _, l, _ in metadata if l == 1)
    total_negative = sum(1 for _, l, _ in metadata if l == 0)
    
    print(f"\n  Positive samples (Hey Orvyn): {total_positive}")
    print(f"  Negative samples (other):     {total_negative}")
    print(f"\n  Per language breakdown:")
    
    for lang, count in sorted(lang_counts.items()):
        bar = "█" * min(count // 3, 40)
        print(f"  {lang:10} {count:4} samples  {bar}")
    
    print(f"\n  Each sample shape: (40, 101)")
    print(f"  = 40 mel bands × 101 time steps = 4,040 features per sample")
    
    # Estimate memory usage
    bytes_per_sample = 40 * 101 * 4  # float32 = 4 bytes
    total_mb = (total_positive + total_negative) * bytes_per_sample / 1024 / 1024
    print(f"\n  Estimated dataset size in memory: {total_mb:.1f} MB")
    print("="*60)
    
    if total_negative == 0:
        print("""
  ⚠️  WARNING: No negative samples yet!
  
  Before training, we need negative samples —
  audio that is NOT "Hey Orvyn" so the model
  learns what to ignore.
  
  Next step: Ask "write the negative samples download script"
  We'll download free speech data from Mozilla Common Voice.
        """)


# ── VISUALIZE ONE SAMPLE ──────────────────────────────────────────────────────
def visualize_sample():
    """
    Shows what a mel spectrogram looks like for one sample.
    This helps you understand what the model is actually learning from.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')  # non-interactive backend for saving
        
        # Find first processed file
        npy_files = list(POSITIVE_PROC_DIR.rglob("*.npy"))
        if not npy_files:
            return
        
        # Load and display
        sample = np.load(str(npy_files[0]))
        
        plt.figure(figsize=(10, 4))
        plt.imshow(
            sample,
            aspect='auto',
            origin='lower',
            cmap='viridis'
        )
        plt.colorbar(label='Log-Mel Energy (normalized)')
        plt.title(f'Mel Spectrogram of "Hey Orvyn"\n{npy_files[0].name}')
        plt.xlabel('Time (10ms steps)')
        plt.ylabel('Mel Frequency Band')
        plt.tight_layout()
        plt.savefig('data/processed/sample_spectrogram.png', dpi=100)
        plt.close()
        
        print("\n✅ Sample spectrogram saved to:")
        print("   data/processed/sample_spectrogram.png")
        print("   Open this file to see what the model learns from!")
        
    except Exception as e:
        pass  # visualization is optional


# ── ENTRY POINT ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("="*60)
    print("  HEY ORVYN - AUDIO PREPROCESSING")
    print("  Converting wav files to mel spectrograms")
    print("="*60)
    
    # Step 1: Create output folders
    create_output_folders()
    
    # Step 2: Process positive samples
    positive_metadata = process_positive_samples()
    
    # Step 3: Process negative samples (if any)
    negative_metadata = process_negative_samples()
    
    # Step 4: Combine and save metadata
    all_metadata = positive_metadata + negative_metadata
    save_metadata(all_metadata)
    
    # Step 5: Print statistics
    print_statistics(all_metadata)
    
    # Step 6: Visualize one sample
    visualize_sample()
    
    print("""
  ✅ Preprocessing complete!
  
  Your data is ready at: data/processed/
  
  Next step: Download negative samples, then train the model.
  Ask: "write the negative samples download script"
    """)