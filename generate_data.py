"""
=============================================================
  Hey Orvyn - Wake Word Data Generation Script
  Calls ElevenLabs + Sarvam AI APIs to generate training data
=============================================================

HOW TO RUN:
  python generate_data.py

WHAT IT DOES:
  1. Calls ElevenLabs API → "Hey Orvyn" in 29 languages, multiple voices
  2. Calls Sarvam AI API  → "Hey Orvyn" in 11 Indian languages, 38 voices
  3. Saves all audio as .wav files in data/raw/positive/<language>/
  4. Shows progress bar for every step
  5. Skips errors automatically and continues

OUTPUT FOLDER STRUCTURE:
  data/raw/positive/
    en/   ← English samples
    hi/   ← Hindi samples
    te/   ← Telugu samples
    ... and so on for all languages
"""

# ── IMPORTS ──────────────────────────────────────────────────────────────────
# os     : interact with the file system (create folders, build paths)
# time   : add small delays between API calls so we don't get rate-limited
# base64 : Sarvam API returns audio as base64 text — we decode it to bytes
# json   : parse API responses (APIs send back JSON formatted text)
# pathlib: modern way to handle file paths on any OS (Windows/Mac/Linux)

import os
import time
import base64
import json
import requests                          # makes HTTP API calls
from pathlib import Path                 # file path handling
from dotenv import load_dotenv           # reads our .env file
from tqdm import tqdm                    # progress bars
import soundfile as sf                   # saves audio files
import numpy as np                       # number arrays
import io                                # in-memory file handling

# ── LOAD API KEYS FROM .env FILE ─────────────────────────────────────────────
# load_dotenv() reads your .env file and makes the keys available
# os.getenv() then fetches each key by name
load_dotenv()
ELEVENLABS_KEY = os.getenv("ELEVENLABS_API_KEY")
SARVAM_KEY     = os.getenv("SARVAM_API_KEY")

# Safety check — if keys are missing, tell the user immediately
if not ELEVENLABS_KEY:
    raise ValueError("❌ ELEVENLABS_API_KEY not found in .env file!")
if not SARVAM_KEY:
    raise ValueError("❌ SARVAM_API_KEY not found in .env file!")

print("✅ API keys loaded successfully")

# ── CONFIGURATION ─────────────────────────────────────────────────────────────
# This is the text we want to convert to speech
# We generate slight variations so the model learns different natural phrasings
WAKE_WORD_TEXTS = [
    "Hey Orvyn",
    "Hey Orvyn.",        # with pause at end
    "Hey, Orvyn",        # with comma — slight pause between words
]

# Output folder — all positive samples go here
OUTPUT_DIR = Path("data/raw/positive")

# ── ELEVENLABS CONFIGURATION ──────────────────────────────────────────────────
#
# WHAT IS eleven_multilingual_v2?
# It's ElevenLabs' best quality multilingual model.
# It supports 29 languages and produces very natural speech.
# We use it because "Hey Orvyn" needs to sound natural in every language.
#
# WHAT ARE VOICE IDs?
# ElevenLabs has a library of pre-made voices. Each has a unique ID.
# We use multiple voices so our model learns "Hey Orvyn" from
# different people — male, female, different accents, ages.
# More voice variety = model works for more real users.

ELEVENLABS_MODEL = "eleven_multilingual_v2"

# These are real ElevenLabs voice IDs from their free voice library
# Each ID is a different person's voice
ELEVENLABS_VOICES = [
    {"id": "21m00Tcm4TlvDq8ikWAM", "name": "Rachel"},      # female, calm
    {"id": "AZnzlk1XvdvUeBnXmlld", "name": "Domi"},        # female, strong
    {"id": "EXAVITQu4vr4xnSDxMaL", "name": "Bella"},       # female, soft
    {"id": "ErXwobaYiN019PkySvjV", "name": "Antoni"},      # male, well-rounded
    {"id": "MF3mGyEYCl7XYWbV9V6O", "name": "Elli"},        # female, emotional
    {"id": "TxGEqnHWrfWFTfGW9XjX", "name": "Josh"},        # male, deep
    {"id": "VR6AewLTigWG4xSOukaG", "name": "Arnold"},      # male, crisp
    {"id": "pNInz6obpgDQGcFmaJgB", "name": "Adam"},        # male, neutral
    {"id": "yoZ06aMxZJJ28mfd3POQ", "name": "Sam"},         # male, raspy
    {"id": "jBpfuIE2acCO8z3wKNLl", "name": "Gigi"},        # female, childlike
]

# All 29 languages ElevenLabs multilingual_v2 supports
# Format: {"code": "language code used by API", "name": "human readable name", "folder": "folder name"}
ELEVENLABS_LANGUAGES = [
    {"code": "en", "name": "English",    "folder": "en"},
    {"code": "es", "name": "Spanish",    "folder": "es"},
    {"code": "fr", "name": "French",     "folder": "fr"},
    {"code": "de", "name": "German",     "folder": "de"},
    {"code": "it", "name": "Italian",    "folder": "it"},
    {"code": "pt", "name": "Portuguese", "folder": "pt"},
    {"code": "ru", "name": "Russian",    "folder": "ru"},
    {"code": "ja", "name": "Japanese",   "folder": "ja"},
    {"code": "ko", "name": "Korean",     "folder": "ko"},
    {"code": "zh", "name": "Mandarin",   "folder": "zh"},
    {"code": "ar", "name": "Arabic",     "folder": "ar"},
    {"code": "hi", "name": "Hindi",      "folder": "hi"},
    {"code": "nl", "name": "Dutch",      "folder": "nl"},
    {"code": "pl", "name": "Polish",     "folder": "pl"},
    {"code": "tr", "name": "Turkish",    "folder": "tr"},
    {"code": "sv", "name": "Swedish",    "folder": "sv"},
    {"code": "id", "name": "Indonesian", "folder": "id"},
    {"code": "tl", "name": "Filipino",   "folder": "tl"},
    {"code": "uk", "name": "Ukrainian",  "folder": "uk"},
    {"code": "el", "name": "Greek",      "folder": "el"},
    {"code": "cs", "name": "Czech",      "folder": "cs"},
    {"code": "fi", "name": "Finnish",    "folder": "fi"},
    {"code": "ro", "name": "Romanian",   "folder": "ro"},
    {"code": "da", "name": "Danish",     "folder": "da"},
    {"code": "hu", "name": "Hungarian",  "folder": "hu"},
    {"code": "no", "name": "Norwegian",  "folder": "no"},
    {"code": "vi", "name": "Vietnamese", "folder": "vi"},
    {"code": "ta", "name": "Tamil",      "folder": "ta"},
    {"code": "ms", "name": "Malay",      "folder": "ms"},
]

# ── SARVAM AI CONFIGURATION ───────────────────────────────────────────────────
#
# WHAT IS bulbul:v3?
# Sarvam's best TTS model, built specifically for Indian languages.
# It understands Indian language phonetics naturally — much better
# than any Western TTS model for these languages.
#
# WHY USE SARVAM FOR INDIC LANGUAGES?
# ElevenLabs handles Hindi and Tamil, but for Telugu, Kannada, Malayalam
# etc. the pronunciation is much more natural with Sarvam's model.
# We use Sarvam for all 11 Indic languages it supports.

SARVAM_MODEL = "bulbul:v3"
SARVAM_API_URL = "https://api.sarvam.ai/text-to-speech"

# All 38 Sarvam voices — looping these gives us massive voice variety
# This means "Hey Orvyn" in Telugu spoken by 38 different people
SARVAM_VOICES = [
    "shubh", "aditya", "ritu", "priya", "neha", "rahul", "pooja",
    "rohan", "simran", "kavya", "amit", "dev", "ishita", "shreya",
    "ratan", "varun", "manan", "sumit", "roopa", "kabir", "aayan",
    "ashutosh", "advait", "amelia", "sophia", "anand", "tanya",
    "tarun", "sunny", "mani", "gokul", "vijay", "shruti", "suhani",
    "mohit", "kavitha", "rehan", "soham", "rupali"
]

# Sarvam's 11 supported TTS languages with their BCP-47 codes
# BCP-47 is a standard way to write language codes: language-COUNTRY
SARVAM_LANGUAGES = [
    {"code": "hi-IN", "name": "Hindi",     "folder": "hi"},
    {"code": "te-IN", "name": "Telugu",    "folder": "te"},
    {"code": "ta-IN", "name": "Tamil",     "folder": "ta"},
    {"code": "kn-IN", "name": "Kannada",   "folder": "kn"},
    {"code": "ml-IN", "name": "Malayalam", "folder": "ml"},
    {"code": "bn-IN", "name": "Bengali",   "folder": "bn"},
    {"code": "mr-IN", "name": "Marathi",   "folder": "mr"},
    {"code": "gu-IN", "name": "Gujarati",  "folder": "gu"},
    {"code": "pa-IN", "name": "Punjabi",   "folder": "pa"},
    {"code": "or-IN", "name": "Odia",      "folder": "or"},
    {"code": "en-IN", "name": "English-IN","folder": "en_in"},
]

# ── HELPER: CREATE FOLDERS ────────────────────────────────────────────────────
def create_language_folders():
    """
    Creates output folders for every language before we start generating.
    
    WHY: If a folder doesn't exist when we try to save a file, Python crashes.
    We create all folders first so saving never fails.
    
    Path.mkdir(parents=True, exist_ok=True) means:
    - parents=True  : create parent folders too if they don't exist
    - exist_ok=True : don't crash if the folder already exists
    """
    all_languages = (
        [l["folder"] for l in ELEVENLABS_LANGUAGES] +
        [l["folder"] for l in SARVAM_LANGUAGES]
    )
    # Remove duplicates (hi appears in both lists)
    all_languages = list(set(all_languages))
    
    for lang_folder in all_languages:
        folder = OUTPUT_DIR / lang_folder
        folder.mkdir(parents=True, exist_ok=True)
    
    print(f"✅ Created {len(all_languages)} language folders in {OUTPUT_DIR}/")


# ── ELEVENLABS API FUNCTION ───────────────────────────────────────────────────
def generate_elevenlabs(text, voice_id, language_code, output_path):
    """
    Calls ElevenLabs API to generate one audio sample.
    
    PARAMETERS:
    - text          : "Hey Orvyn" or a variation
    - voice_id      : which voice to use (e.g. "21m00Tcm4TlvDq8ikWAM")
    - language_code : which language (e.g. "te" for Telugu)
    - output_path   : where to save the .wav file
    
    HOW THE API WORKS:
    1. We send a POST request with our text + settings
    2. ElevenLabs returns raw audio bytes (mp3 format)
    3. We save those bytes to a .wav file
    
    RETURNS: True if successful, False if failed
    """
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    
    headers = {
        "xi-api-key": ELEVENLABS_KEY,        # our API key for authentication
        "Content-Type": "application/json",   # we're sending JSON data
        "Accept": "audio/mpeg",               # we want MP3 audio back
    }
    
    # The request body — settings for this generation
    payload = {
        "text": text,
        "model_id": ELEVENLABS_MODEL,
        "language_code": language_code,
        "voice_settings": {
            "stability": 0.5,         # 0=very variable, 1=very consistent
            "similarity_boost": 0.75, # how closely to match the original voice
            "style": 0.0,             # no extra style exaggeration
            "use_speaker_boost": True # enhance speaker clarity
        },
        "output_format": "pcm_16000" # 16kHz PCM — perfect for our model
    }
    
    try:
        # Make the API call with a 30 second timeout
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        
        # Check if the call succeeded (HTTP 200 = success)
        if response.status_code == 200:
            # Save the raw audio bytes directly as a .wav file
            with open(output_path, "wb") as f:
                f.write(response.content)
            return True
        else:
            # API returned an error — print it so we know what went wrong
            print(f"\n  ⚠️  ElevenLabs error {response.status_code}: {response.text[:100]}")
            return False
            
    except Exception as e:
        print(f"\n  ⚠️  ElevenLabs exception: {str(e)[:100]}")
        return False


# ── SARVAM AI API FUNCTION ────────────────────────────────────────────────────
def generate_sarvam(text, voice, language_code, output_path):
    """
    Calls Sarvam AI API to generate one audio sample.
    
    HOW SARVAM IS DIFFERENT FROM ELEVENLABS:
    - Sarvam returns audio as base64 encoded text (not raw bytes)
    - base64 is a way to represent binary data as text characters
    - We must decode it back to bytes before saving
    
    Think of base64 like this: imagine converting a photo to a very long
    string of letters and numbers. That's base64. We reverse that process.
    
    RETURNS: True if successful, False if failed
    """
    headers = {
        "api-subscription-key": SARVAM_KEY,  # Sarvam uses this header for auth
        "Content-Type": "application/json",
    }
    
    payload = {
        "inputs": [text],              # list of texts to convert
        "target_language_code": language_code,  # e.g. "te-IN" for Telugu
        "speaker": voice,              # which voice to use
        "model": SARVAM_MODEL,         # "bulbul:v3"
                            # normal pitch (range: -0.75 to 0.75)
        "pace": 1.0,                   # normal speed (range: 0.5 to 2.0)
                       # normal volume
        "speech_sample_rate": 16000,   # 16kHz — exactly what our model needs
        "enable_preprocessing": True,  # auto-fix text formatting
        "eng_interpolation_wt": 0,     # don't mix English into Indian languages
    }
    
    try:
        response = requests.post(
            SARVAM_API_URL,
            headers=headers,
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            
            # Sarvam returns: {"audios": ["base64encodedaudiostring..."]}
            # We get the first (and only) audio from the list
            audio_b64 = data["audios"][0]
            
            # Decode base64 string → raw audio bytes
            # This is like un-converting the photo back from text
            audio_bytes = base64.b64decode(audio_b64)
            
            # The bytes are raw PCM audio (just numbers, no header)
            # We convert to numpy array and save properly as .wav
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
            
            # Convert int16 (-32768 to 32767) to float32 (-1.0 to 1.0)
            # This is the standard format for audio processing
            audio_float = audio_array.astype(np.float32) / 32768.0
            
            # Save as .wav file at 16kHz sample rate
            sf.write(str(output_path), audio_float, 16000)
            return True
        else:
            print(f"\n  ⚠️  Sarvam error {response.status_code}: {response.text[:100]}")
            return False
            
    except Exception as e:
        print(f"\n  ⚠️  Sarvam exception: {str(e)[:100]}")
        return False


# ── MAIN GENERATION FUNCTION ──────────────────────────────────────────────────
def generate_elevenlabs_dataset():
    """
    Loops through all ElevenLabs languages × voices × text variations
    and generates audio files for each combination.
    
    TOTAL FILES = languages × voices × text_variations
                = 29 × 10 × 3 = 870 files
    
    Each file is named: {language}_{voice_name}_{text_index}_{timestamp}.wav
    Example: en_Rachel_0_1234567890.wav
    """
    print("\n" + "="*60)
    print("  GENERATING ELEVENLABS DATA")
    print("="*60)
    
    total = len(ELEVENLABS_LANGUAGES) * len(ELEVENLABS_VOICES) * len(WAKE_WORD_TEXTS)
    success_count = 0
    fail_count = 0
    
    # tqdm creates a progress bar that shows how far along we are
    with tqdm(total=total, desc="ElevenLabs", unit="file") as pbar:
        
        for lang in ELEVENLABS_LANGUAGES:
            lang_folder = OUTPUT_DIR / lang["folder"]
            
            for voice in ELEVENLABS_VOICES:
                for i, text in enumerate(WAKE_WORD_TEXTS):
                    
                    # Build the output filename
                    # timestamp makes each filename unique
                    timestamp = int(time.time() * 1000)
                    filename = f"{lang['folder']}_{voice['name']}_{i}_{timestamp}.wav"
                    output_path = lang_folder / filename
                    
                    # Skip if file already exists (allows resuming if interrupted)
                    if output_path.exists():
                        pbar.update(1)
                        success_count += 1
                        continue
                    
                    # Call the API
                    success = generate_elevenlabs(
                        text=text,
                        voice_id=voice["id"],
                        language_code=lang["code"],
                        output_path=output_path
                    )
                    
                    if success:
                        success_count += 1
                    else:
                        fail_count += 1
                    
                    pbar.update(1)
                    pbar.set_postfix({
                        "lang": lang["name"],
                        "voice": voice["name"],
                        "✅": success_count,
                        "❌": fail_count
                    })
                    
                    # Small delay between calls to avoid hitting rate limits
                    # Rate limit = max number of API calls per second
                    # Without this delay, the API might reject our requests
                    time.sleep(0.5)
    
    print(f"\n  ElevenLabs done: {success_count} success, {fail_count} failed")
    return success_count, fail_count


def generate_sarvam_dataset():
    """
    Loops through all Sarvam languages × voices × text variations.
    
    TOTAL FILES = languages × voices × text_variations
                = 11 × 38 × 3 = 1,254 files
    
    Combined with ElevenLabs: 870 + 1,254 = 2,124 positive samples total
    """
    print("\n" + "="*60)
    print("  GENERATING SARVAM AI DATA")
    print("="*60)
    
    total = len(SARVAM_LANGUAGES) * len(SARVAM_VOICES) * len(WAKE_WORD_TEXTS)
    success_count = 0
    fail_count = 0
    
    with tqdm(total=total, desc="Sarvam AI", unit="file") as pbar:
        
        for lang in SARVAM_LANGUAGES:
            lang_folder = OUTPUT_DIR / lang["folder"]
            
            for voice in SARVAM_VOICES:
                for i, text in enumerate(WAKE_WORD_TEXTS):
                    
                    timestamp = int(time.time() * 1000)
                    filename = f"{lang['folder']}_{voice}_{i}_{timestamp}.wav"
                    output_path = lang_folder / filename
                    
                    if output_path.exists():
                        pbar.update(1)
                        success_count += 1
                        continue
                    
                    success = generate_sarvam(
                        text=text,
                        voice=voice,
                        language_code=lang["code"],
                        output_path=output_path
                    )
                    
                    if success:
                        success_count += 1
                    else:
                        fail_count += 1
                    
                    pbar.update(1)
                    pbar.set_postfix({
                        "lang": lang["name"],
                        "voice": voice,
                        "✅": success_count,
                        "❌": fail_count
                    })
                    
                    # Sarvam is slightly slower so we wait a bit longer
                    time.sleep(0.3)
    
    print(f"\n  Sarvam done: {success_count} success, {fail_count} failed")
    return success_count, fail_count


def print_summary():
    """
    After generation, count all files and print a summary per language.
    This lets us verify everything was saved correctly.
    """
    print("\n" + "="*60)
    print("  DATASET SUMMARY")
    print("="*60)
    
    total_files = 0
    
    # Walk through every language folder and count .wav files
    for lang_folder in sorted(OUTPUT_DIR.iterdir()):
        if lang_folder.is_dir():
            wav_files = list(lang_folder.glob("*.wav"))
            count = len(wav_files)
            total_files += count
            # Print a bar chart style display
            bar = "█" * min(count // 5, 40)
            print(f"  {lang_folder.name:8} {count:4} files  {bar}")
    
    print(f"\n  TOTAL: {total_files} positive samples generated")
    print(f"  Saved in: {OUTPUT_DIR.absolute()}")
    print("="*60)


# ── ENTRY POINT ───────────────────────────────────────────────────────────────
# This block only runs when you execute this file directly
# (not when another script imports it)
# It's Python convention — every script has this at the bottom

if __name__ == "__main__":
    print("="*60)
    print("  HEY ORVYN - DATA GENERATION")
    print("  Generating wake word samples in 40 languages")
    print("="*60)
    
    # Step 1: Create all folders
    create_language_folders()
    
    # Step 2: Generate ElevenLabs data (29 languages)
    el_success, el_fail = generate_elevenlabs_dataset()
    
    # Step 3: Generate Sarvam AI data (11 Indian languages)
    sv_success, sv_fail = generate_sarvam_dataset()
    
    # Step 4: Print final summary
    print_summary()
    
    print(f"""
  ✅ Generation complete!
  
  Total generated : {el_success + sv_success} files
  Total failed    : {el_fail + sv_fail} files
  
  Next step: Run the preprocessing script to convert
  all .wav files to mel spectrograms for training.
  
  Ask: "write the preprocessing script"
    """)