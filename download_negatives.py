"""
=============================================================
  Hey Orvyn - Negative Samples Download Script
  Downloads free speech data for "not wake word" training
=============================================================

HOW TO RUN:
  python download_negatives.py

WHAT IT DOES:
  Downloads audio from these FREE sources:
  1. Mozilla Common Voice clips (multiple languages)
  2. Generates hard negatives using TTS 
     (words that sound similar to "Hey Orvyn")

WHY DO WE NEED NEGATIVES?
  Right now our model only knows what "Hey Orvyn" sounds like.
  Without negative examples it would trigger on EVERYTHING.
  
  Think of it like teaching a child to recognize a cat:
  - Positive: "this IS a cat" (1709 samples) ✅
  - Negative: "this is NOT a cat" (dog, car, tree...) ← we need this
  
  Without negatives → model says "Hey Orvyn!" for every sound.
  With negatives → model learns to ignore everything else.

HOW MANY NEGATIVES DO WE NEED?
  Industry standard: 10x more negatives than positives.
  We have 1,709 positives → we need ~17,000 negatives.
  
  We'll get:
  - ~10,000 from Common Voice (real human speech)
  - ~2,000 hard negatives (similar-sounding words via TTS)
  - ~5,000 from free speech datasets
"""

import os
import requests
import zipfile
import tarfile
import shutil
import random
import time
import numpy as np
import soundfile as sf
import torch
import torchaudio.transforms as T
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

# ── CONFIGURATION ─────────────────────────────────────────────────────────────
NEGATIVE_RAW_DIR  = Path("data/raw/negative")
NEGATIVE_PROC_DIR = Path("data/processed/negative")
SAMPLE_RATE       = 16000
MAX_SAMPLES       = 17000  # total negatives we want

# Hard negatives — words that sound phonetically similar to "Hey Orvyn"
# These are the most important negatives because they're the hardest
# for the model to distinguish. If the model can reject these,
# it will reject everything else easily.
HARD_NEGATIVE_TEXTS = [
    # Similar to "Hey"
    "Hey",
    "Hey there",
    "Hey you",
    "Hey now",
    "Hi there",
    "Hello",
    "Hello there",
    
    # Similar to "Orvyn"
    "Orion",
    "Kevin",
    "Irving",
    "Irvin",
    "Orvin",
    "Arvin",
    "Calvin",
    "Gavin",
    "Devin",
    "Ivan",
    
    # Similar full phrases
    "Hey Kevin",
    "Hey Irvin",
    "Hey Orion",
    "Hey Arvin",
    "Hello Orvyn",
    "Hi Orvyn",
    "OK Orvyn",
    "Okay Orvyn",
    
    # Common voice commands that might trigger false accepts
    "Hey Google",
    "Hey Siri",
    "Alexa",
    "OK Google",
    "Hey Cortana",
]

# ── CREATE FOLDERS ─────────────────────────────────────────────────────────────
def create_folders():
    NEGATIVE_RAW_DIR.mkdir(parents=True, exist_ok=True)
    NEGATIVE_PROC_DIR.mkdir(parents=True, exist_ok=True)
    (NEGATIVE_RAW_DIR / "common_voice").mkdir(exist_ok=True)
    (NEGATIVE_RAW_DIR / "hard_negatives").mkdir(exist_ok=True)
    (NEGATIVE_RAW_DIR / "free_speech").mkdir(exist_ok=True)
    print("✅ Negative sample folders created")


# ── METHOD 1: DOWNLOAD FREE SPEECH DATASETS ───────────────────────────────────
def download_librispeech_sample():
    """
    Downloads a small subset of LibriSpeech — a free, high quality
    English speech dataset from OpenSLR.
    
    WHAT IS LIBRISPEECH?
    1000 hours of English audiobook recordings. 
    Completely free, no login needed.
    We download just the 'test-clean' subset (~340MB) which has
    ~2620 utterances from 40 speakers.
    
    WHY USE THIS AS NEGATIVES?
    These are real human voices saying real English sentences —
    none of which are "Hey Orvyn". Perfect negative examples.
    """
    print("\n" + "="*60)
    print("  DOWNLOADING LIBRISPEECH (English speech)")
    print("="*60)
    
    url = "https://www.openslr.org/resources/12/test-clean.tar.gz"
    output_dir = NEGATIVE_RAW_DIR / "free_speech" / "librispeech"
    
    if output_dir.exists() and len(list(output_dir.rglob("*.flac"))) > 100:
        count = len(list(output_dir.rglob("*.flac")))
        print(f"  ✅ Already downloaded ({count} files)")
        return count
    
    output_dir.mkdir(parents=True, exist_ok=True)
    tar_path = NEGATIVE_RAW_DIR / "free_speech" / "librispeech.tar.gz"
    
    print(f"  Downloading from OpenSLR (~340MB)...")
    print(f"  This may take 5-10 minutes depending on your connection")
    
    try:
        response = requests.get(url, stream=True, timeout=60)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(tar_path, 'wb') as f, tqdm(
            total=total_size,
            unit='B',
            unit_scale=True,
            desc="Downloading"
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))
        
        print("  Extracting...")
        with tarfile.open(tar_path, 'r:gz') as tar:
            tar.extractall(output_dir)
        
        tar_path.unlink()  # delete the tar file to save space
        
        count = len(list(output_dir.rglob("*.flac")))
        print(f"  ✅ LibriSpeech downloaded: {count} audio files")
        return count
        
    except Exception as e:
        print(f"  ⚠️  LibriSpeech download failed: {e}")
        print(f"  Skipping — will use other sources")
        return 0


def download_cv_clips():
    """
    Downloads Mozilla Common Voice validated clips.
    
    Common Voice is a crowd-sourced multilingual speech dataset.
    Millions of recordings from real people worldwide.
    We use it because:
    1. Multiple languages (our model needs to ignore speech in all languages)
    2. Real microphone recordings (not studio quality — matches real use)
    3. Completely free, no API key needed
    
    We download pre-extracted clips from a public mirror.
    """
    print("\n" + "="*60)
    print("  DOWNLOADING COMMON VOICE CLIPS")
    print("="*60)
    
    # We'll use the CommonVoice delta clips which are smaller
    # These are from a public dataset mirror on HuggingFace
    urls = [
        {
            "name": "cv_en_sample",
            "url": "https://huggingface.co/datasets/mozilla-foundation/common_voice_11_0/resolve/main/data/en/validated_sentences.tsv",
            "description": "English Common Voice"
        }
    ]
    
    # Actually download from a simpler source — free speech commands
    return download_speech_commands()


def download_speech_commands():
    """
    Downloads Google Speech Commands dataset.
    35 common English words spoken by thousands of people.
    
    WHY PERFECT FOR NEGATIVES:
    Words like "yes", "no", "stop", "go", "up", "down" etc.
    None of these are "Hey Orvyn" — perfect negatives.
    The dataset has ~105,000 clips of 1 second each.
    We only download a small subset.
    """
    print("\n" + "="*60)
    print("  DOWNLOADING GOOGLE SPEECH COMMANDS")
    print("="*60)
    
    url = "http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz"
    output_dir = NEGATIVE_RAW_DIR / "free_speech" / "speech_commands"
    
    if output_dir.exists() and len(list(output_dir.rglob("*.wav"))) > 1000:
        count = len(list(output_dir.rglob("*.wav")))
        print(f"  ✅ Already downloaded ({count} files)")
        return count
    
    output_dir.mkdir(parents=True, exist_ok=True)
    tar_path = NEGATIVE_RAW_DIR / "free_speech" / "speech_commands.tar.gz"
    
    print(f"  Downloading Google Speech Commands (~2.3GB)...")
    print(f"  This will take 10-20 minutes.")
    print(f"  You can let this run in background.")
    
    try:
        response = requests.get(url, stream=True, timeout=120)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(tar_path, 'wb') as f, tqdm(
            total=total_size,
            unit='B',
            unit_scale=True,
            desc="Downloading"
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))
        
        print("  Extracting (this takes a few minutes)...")
        with tarfile.open(tar_path, 'r:gz') as tar:
            tar.extractall(output_dir)
        
        tar_path.unlink()
        
        count = len(list(output_dir.rglob("*.wav")))
        print(f"  ✅ Speech Commands downloaded: {count} audio files")
        return count
        
    except Exception as e:
        print(f"  ⚠️  Speech Commands download failed: {e}")
        return 0


# ── METHOD 2: GENERATE HARD NEGATIVES VIA SARVAM TTS ─────────────────────────
def generate_hard_negatives_sarvam():
    """
    Generates phonetically similar words using Sarvam AI TTS.
    
    WHY HARD NEGATIVES MATTER:
    Easy negatives: "the weather is nice today" → model easily rejects
    Hard negatives: "Hey Kevin", "Hey Orion" → model might confuse with "Hey Orvyn"
    
    Training on hard negatives makes the model much more precise.
    This is called "hard negative mining" in ML — a key technique
    used by all production wake word systems including Picovoice.
    
    We generate these in Hindi, Telugu, Tamil, English because
    those are our primary target languages.
    """
    SARVAM_KEY = os.getenv("SARVAM_API_KEY")
    if not SARVAM_KEY:
        print("  ⚠️  No Sarvam key found — skipping hard negatives")
        return 0
    
    print("\n" + "="*60)
    print("  GENERATING HARD NEGATIVES (similar-sounding phrases)")
    print("="*60)
    
    output_dir = NEGATIVE_RAW_DIR / "hard_negatives"
    
    # Languages and voices to use for hard negatives
    languages = [
        {"code": "hi-IN", "folder": "hi"},
        {"code": "te-IN", "folder": "te"},
        {"code": "ta-IN", "folder": "ta"},
        {"code": "en-IN", "folder": "en_in"},
    ]
    
    # Use just 5 voices for hard negatives (enough variety)
    voices = ["shubh", "priya", "rahul", "neha", "aditya"]
    
    total = len(HARD_NEGATIVE_TEXTS) * len(languages) * len(voices)
    success = 0
    failed  = 0
    
    with tqdm(total=total, desc="Hard negatives", unit="file") as pbar:
        for lang in languages:
            lang_dir = output_dir / lang["folder"]
            lang_dir.mkdir(exist_ok=True)
            
            for voice in voices:
                for i, text in enumerate(HARD_NEGATIVE_TEXTS):
                    
                    timestamp = int(time.time() * 1000)
                    filename  = f"neg_{lang['folder']}_{voice}_{i}_{timestamp}.wav"
                    out_path  = lang_dir / filename
                    
                    if out_path.exists():
                        success += 1
                        pbar.update(1)
                        continue
                    
                    try:
                        response = requests.post(
                            "https://api.sarvam.ai/text-to-speech",
                            headers={
                                "api-subscription-key": SARVAM_KEY,
                                "Content-Type": "application/json"
                            },
                            json={
                                "inputs": [text],
                                "target_language_code": lang["code"],
                                "speaker": voice,
                                "model": "bulbul:v3",
                                "pace": 1.0,
                                "speech_sample_rate": 16000,
                                "enable_preprocessing": True,
                            },
                            timeout=30
                        )
                        
                        if response.status_code == 200:
                            import base64
                            data      = response.json()
                            audio_b64 = data["audios"][0]
                            audio_bytes = base64.b64decode(audio_b64)
                            audio_array = np.frombuffer(
                                audio_bytes, dtype=np.int16
                            ).astype(np.float32) / 32768.0
                            sf.write(str(out_path), audio_array, 16000)
                            success += 1
                        else:
                            failed += 1
                            
                    except Exception as e:
                        failed += 1
                    
                    pbar.update(1)
                    pbar.set_postfix({"✅": success, "❌": failed})
                    time.sleep(0.3)
    
    print(f"\n  Hard negatives done: {success} success, {failed} failed")
    return success


# ── METHOD 3: GENERATE HARD NEGATIVES VIA ELEVENLABS ─────────────────────────
def generate_hard_negatives_elevenlabs():
    """
    Uses ElevenLabs to generate hard negatives in global languages.
    Only runs if you have ElevenLabs credits.
    """
    ELEVENLABS_KEY = os.getenv("ELEVENLABS_API_KEY")
    if not ELEVENLABS_KEY:
        print("  ⚠️  No ElevenLabs key — skipping")
        return 0
    
    print("\n" + "="*60)
    print("  GENERATING HARD NEGATIVES (ElevenLabs — global langs)")
    print("="*60)
    
    output_dir = NEGATIVE_RAW_DIR / "hard_negatives" / "global"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Just use 3 voices and 3 languages to save credits
    voices = [
        {"id": "21m00Tcm4TlvDq8ikWAM", "name": "Rachel"},
        {"id": "TxGEqnHWrfWFTfGW9XjX", "name": "Josh"},
        {"id": "pNInz6obpgDQGcFmaJgB", "name": "Adam"},
    ]
    languages = ["en", "es", "fr"]
    
    # Only generate the most important hard negatives
    key_phrases = [
        "Hey Kevin", "Hey Orion", "Hey Siri",
        "Hello", "Hey Google", "OK Google",
        "Hey there", "Hi there", "Alexa",
    ]
    
    success = 0
    failed  = 0
    total   = len(key_phrases) * len(voices) * len(languages)
    
    with tqdm(total=total, desc="ElevenLabs negatives", unit="file") as pbar:
        for lang in languages:
            for voice in voices:
                for i, text in enumerate(key_phrases):
                    
                    timestamp = int(time.time() * 1000)
                    filename  = f"neg_{lang}_{voice['name']}_{i}_{timestamp}.wav"
                    out_path  = output_dir / filename
                    
                    if out_path.exists():
                        success += 1
                        pbar.update(1)
                        continue
                    
                    try:
                        response = requests.post(
                            f"https://api.elevenlabs.io/v1/text-to-speech/{voice['id']}",
                            headers={
                                "xi-api-key": ELEVENLABS_KEY,
                                "Content-Type": "application/json",
                                "Accept": "audio/mpeg"
                            },
                            json={
                                "text": text,
                                "model_id": "eleven_multilingual_v2",
                                "language_code": lang,
                                "voice_settings": {
                                    "stability": 0.5,
                                    "similarity_boost": 0.75,
                                },
                                "output_format": "pcm_16000"
                            },
                            timeout=30
                        )
                        
                        if response.status_code == 200:
                            with open(out_path, 'wb') as f:
                                f.write(response.content)
                            success += 1
                        else:
                            failed += 1
                            
                    except Exception:
                        failed += 1
                    
                    pbar.update(1)
                    time.sleep(0.5)
    
    print(f"\n  ElevenLabs negatives: {success} success, {failed} failed")
    return success


# ── PREPROCESS NEGATIVES ──────────────────────────────────────────────────────
def preprocess_negatives():
    """
    Converts all negative wav files to mel spectrograms.
    Same preprocessing pipeline as positive samples.
    Saves to data/processed/negative/
    """
    print("\n" + "="*60)
    print("  PREPROCESSING NEGATIVE SAMPLES")
    print("="*60)
    
    mel_transform = T.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=400,
        hop_length=160,
        n_mels=40,
        f_min=80,
        f_max=8000,
        power=2.0,
    )
    
    # Find all wav and flac files in negative folder
    all_files = (
        list(NEGATIVE_RAW_DIR.rglob("*.wav")) +
        list(NEGATIVE_RAW_DIR.rglob("*.flac"))
    )
    
    print(f"  Found {len(all_files)} negative audio files")
    
    if len(all_files) == 0:
        print("  ⚠️  No negative files to process")
        return []
    
    # Limit to MAX_SAMPLES to avoid processing too many
    random.shuffle(all_files)
    all_files = all_files[:MAX_SAMPLES]
    print(f"  Processing {len(all_files)} files (max {MAX_SAMPLES})")
    
    metadata = []
    success  = 0
    failed   = 0
    
    with tqdm(total=len(all_files), desc="Processing negatives") as pbar:
        for audio_path in all_files:
            
            # Build output path
            rel_path = audio_path.relative_to(NEGATIVE_RAW_DIR)
            npy_path = NEGATIVE_PROC_DIR / rel_path.with_suffix('.npy')
            npy_path.parent.mkdir(parents=True, exist_ok=True)
            
            if npy_path.exists():
                metadata.append((str(npy_path), 0, "negative"))
                success += 1
                pbar.update(1)
                continue
            
            try:
                # Load audio
                audio_np, sr = sf.read(str(audio_path))
                
                # Stereo to mono
                if len(audio_np.shape) > 1:
                    audio_np = audio_np.mean(axis=1)
                
                audio_np = audio_np.astype(np.float32)
                waveform = torch.tensor(audio_np).unsqueeze(0)
                
                # Resample to 16kHz
                if sr != SAMPLE_RATE:
                    resampler = T.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)
                    waveform = resampler(waveform)
                
                # Pad or crop to 1 second
                max_len = SAMPLE_RATE
                if waveform.shape[1] < max_len:
                    padding = max_len - waveform.shape[1]
                    waveform = torch.nn.functional.pad(waveform, (0, padding))
                else:
                    waveform = waveform[:, :max_len]
                
                # Mel spectrogram
                mel_spec = mel_transform(waveform).squeeze(0)
                mel_spec = torch.log(mel_spec + 1e-9)
                
                # Normalize
                mean = mel_spec.mean()
                std  = mel_spec.std()
                if std > 0:
                    mel_spec = (mel_spec - mean) / std
                
                # Save
                np.save(str(npy_path), mel_spec.numpy())
                metadata.append((str(npy_path), 0, "negative"))
                success += 1
                
            except Exception as e:
                failed += 1
            
            pbar.update(1)
            pbar.set_postfix({"✅": success, "❌": failed})
    
    print(f"\n  Negatives preprocessed: {success} success, {failed} failed")
    return metadata


# ── UPDATE METADATA CSV ───────────────────────────────────────────────────────
def update_metadata(negative_metadata):
    """
    Adds negative samples to the existing metadata.csv
    that already contains our 1,709 positive samples.
    """
    import csv
    
    metadata_file = Path("data/processed/metadata.csv")
    
    # Read existing metadata (positives)
    existing = []
    if metadata_file.exists():
        with open(metadata_file, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            existing = list(reader)
    
    # Combine positives + negatives
    all_metadata = existing + negative_metadata
    
    # Write back
    with open(metadata_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['filepath', 'label', 'language'])
        writer.writerows(all_metadata)
    
    positives = sum(1 for _, l, _ in all_metadata if l == '1' or l == 1)
    negatives = sum(1 for _, l, _ in all_metadata if l == '0' or l == 0)
    
    print(f"\n✅ Metadata updated:")
    print(f"   Positive samples: {positives}")
    print(f"   Negative samples: {negatives}")
    print(f"   Total:            {len(all_metadata)}")
    print(f"   Ratio:            1:{negatives//max(positives,1)} (pos:neg)")


# ── ENTRY POINT ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("="*60)
    print("  HEY ORVYN - NEGATIVE SAMPLES")
    print("  Downloading non-wake-word speech for training")
    print("="*60)
    print(f"""
  WHY WE NEED THIS:
  We have 1,709 positive samples (Hey Orvyn).
  We need ~17,000 negative samples (everything else).
  Without negatives, the model triggers on all speech.
  
  SOURCES:
  1. Google Speech Commands (~105,000 free clips)
  2. Hard negatives via Sarvam TTS (similar-sounding words)
  3. Hard negatives via ElevenLabs (if credits available)
    """)
    
    # Step 1: Create folders
    create_folders()
    
    # Step 2: Download Google Speech Commands (big download ~2.3GB)
    speech_count = download_speech_commands()
    
    # Step 3: Generate hard negatives with Sarvam
    sarvam_count = generate_hard_negatives_sarvam()
    
    # Step 4: Generate hard negatives with ElevenLabs (if credits)
    eleven_count = generate_hard_negatives_elevenlabs()
    
    # Step 5: Preprocess all negatives
    negative_metadata = preprocess_negatives()
    
    # Step 6: Update metadata.csv
    update_metadata(negative_metadata)
    
    print(f"""
  ✅ Negative samples complete!
  
  Speech Commands : {speech_count} files
  Sarvam negatives: {sarvam_count} files  
  ElevenLabs neg  : {eleven_count} files
  Total processed : {len(negative_metadata)} files
  
  You now have both positive AND negative samples.
  Next step: Train the model!
  Ask: "write the training script"
    """)