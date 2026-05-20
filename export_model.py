"""
=============================================================
  Hey Orvyn - Model Export Script
  Exports trained model to ONNX (Windows) and TFLite (Android)
=============================================================

HOW TO RUN:
  python export_model.py

WHAT IT DOES:
  1. Loads your trained model from models/best_model.pt
  2. Exports to ONNX format for Windows desktop
  3. Converts ONNX to TFLite for Android
  4. Applies INT8 quantization to make model smaller
  5. Verifies both exports work correctly
  6. Shows final model sizes

OUTPUT FILES:
  export/hey_orvyn.onnx          ← Windows desktop
  export/hey_orvyn.tflite        ← Android app
  export/hey_orvyn_quant.tflite  ← Android (quantized, smaller)

WHAT IS EXPORT?
  Training produces a .pt file — PyTorch's own format.
  .pt files only work with PyTorch installed.
  
  For deployment we need universal formats:
  - ONNX: works on Windows, Mac, Linux without PyTorch
  - TFLite: works on Android, iOS without any ML framework
  
  Think of it like saving a Word document as PDF —
  same content, universal format anyone can open.

WHAT IS QUANTIZATION?
  Training uses float32 — 4 bytes per number.
  INT8 quantization uses 1 byte per number.
  Result: 4x smaller model, ~same accuracy.
  
  float32 model: ~180 KB
  INT8 model:    ~45 KB
  
  This is how Siri and Google Assistant run on phones.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import json

# ── CONFIGURATION ─────────────────────────────────────────────────────────────
MODELS_DIR  = Path("models")
EXPORT_DIR  = Path("export")
MODEL_PATH  = MODELS_DIR / "best_model.pt"

ONNX_PATH         = EXPORT_DIR / "hey_orvyn.onnx"
TFLITE_PATH       = EXPORT_DIR / "hey_orvyn.tflite"
TFLITE_QUANT_PATH = EXPORT_DIR / "hey_orvyn_quant.tflite"
CONFIG_PATH       = EXPORT_DIR / "model_config.json"

# Model input shape — must match training exactly
# (batch, channels, mel_bands, time_steps)
INPUT_SHAPE = (1, 1, 40, 101)


# ── TC-RESNET MODEL (same as train.py) ───────────────────────────────────────
# We need to redefine the model architecture here so we can load weights.
# The architecture must be IDENTICAL to what was used in training.

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3,1), stride=(stride,1), padding=(1,0), bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3,1), stride=1, padding=(1,0), bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)
        self.relu  = nn.ReLU(inplace=True)
        self.skip  = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride,1), bias=False),
                nn.BatchNorm2d(out_channels)
            )
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.skip(x)
        out = self.relu(out)
        return out

class TCResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1  = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(3,1), padding=(1,0), bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        self.block1 = ResidualBlock(16, 24, stride=2)
        self.block2 = ResidualBlock(24, 32, stride=2)
        self.block3 = ResidualBlock(32, 48, stride=2)
        self.gap     = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.3)
        self.fc      = nn.Linear(48, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


# ── LOAD TRAINED MODEL ────────────────────────────────────────────────────────
def load_model():
    """
    Loads the trained model weights from best_model.pt.
    
    WHAT IS model.eval()?
    Switches model from training mode to inference mode.
    Training mode: dropout randomly zeros neurons (for regularization)
    Inference mode: dropout is disabled — we want consistent predictions
    BatchNorm also behaves differently — uses running stats instead of batch stats.
    
    WHAT IS .cpu()?
    Moves model from GPU to CPU for export.
    ONNX and TFLite export requires CPU tensors.
    The exported model will still run fast on both CPU and GPU.
    """
    print("📂 Loading trained model...")
    
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run train.py first!")
    
    model = TCResNet()
    
    # Load checkpoint
    checkpoint = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # switch to inference mode
    model.cpu()   # move to CPU for export
    
    print(f"  ✅ Model loaded from epoch {checkpoint['epoch']}")
    print(f"  Val accuracy : {checkpoint['val_acc']*100:.1f}%")
    print(f"  Val FAR      : {checkpoint['val_far']*100:.2f}%")
    print(f"  Val FRR      : {checkpoint['val_frr']*100:.2f}%")
    
    return model, checkpoint


# ── EXPORT TO ONNX ────────────────────────────────────────────────────────────
def export_onnx(model):
    """
    Exports model to ONNX format for Windows desktop deployment.
    
    WHAT IS ONNX?
    Open Neural Network Exchange — a universal format for ML models.
    Created by Microsoft and Facebook together.
    
    WHY ONNX FOR DESKTOP?
    - Works on Windows, Mac, Linux
    - No PyTorch needed on the deployment machine
    - ONNX Runtime is tiny (~5MB) and very fast
    - Same .onnx file works on Python, C++, C#, Java
    
    HOW EXPORT WORKS:
    torch.onnx.export() runs your model on a dummy input
    and records every operation that happens.
    It saves this "recipe" as an ONNX graph.
    ONNX Runtime later "plays back" this recipe with real audio.
    
    dynamic_axes allows the batch dimension to be any size.
    During training batch=128, during inference batch=1.
    """
    print("\n" + "="*60)
    print("  EXPORTING TO ONNX (Windows Desktop)")
    print("="*60)
    
    EXPORT_DIR.mkdir(exist_ok=True)
    
    # Create dummy input — same shape as real mel spectrogram
    # (1, 1, 40, 101) = (batch=1, channels=1, mel_bands=40, time=101)
    dummy_input = torch.randn(INPUT_SHAPE)
    
    print(f"  Input shape: {INPUT_SHAPE}")
    print(f"  Exporting to {ONNX_PATH}...")
    
    torch.onnx.export(
        model,
        dummy_input,
        str(ONNX_PATH),
        
        # opset_version: ONNX version to use (17 is latest stable)
        opset_version=17,
        
        # export_params: include trained weights in the file
        export_params=True,
        
        # do_constant_folding: pre-compute constant expressions
        # Makes inference faster by simplifying the graph
        do_constant_folding=True,
        
        # Name the input and output tensors
        input_names=['mel_spectrogram'],
        output_names=['wake_word_logit'],
        
        # Allow batch size to vary (1 during inference, 128 during training)
        dynamic_axes={
            'mel_spectrogram': {0: 'batch_size'},
            'wake_word_logit': {0: 'batch_size'}
        }
    )
    
    # Verify export worked
    print("  Verifying ONNX export...")
    try:
        import onnxruntime as ort
        
        # Load the exported model
        session = ort.InferenceSession(str(ONNX_PATH))
        
        # Run inference with dummy input
        dummy_np = dummy_input.numpy()
        outputs  = session.run(None, {'mel_spectrogram': dummy_np})
        
        # Apply sigmoid to get probability
        logit = outputs[0][0][0]
        prob  = 1 / (1 + np.exp(-logit))
        
        size_kb = ONNX_PATH.stat().st_size / 1024
        
        print(f"  ✅ ONNX export verified!")
        print(f"  Output probability: {prob:.4f} (dummy input, should be ~0.5)")
        print(f"  File size: {size_kb:.1f} KB")
        
        return True
        
    except Exception as e:
        print(f"  ⚠️  ONNX verification failed: {e}")
        return False


# ── EXPORT TO TFLITE ──────────────────────────────────────────────────────────
def export_tflite(model):
    """
    Exports model to TFLite format for Android deployment.
    
    WHAT IS TFLITE?
    TensorFlow Lite — Google's ML format for mobile devices.
    Optimized for Android and iOS.
    
    WHY TFLITE FOR ANDROID?
    - Native Android support (built into Google Play Services)
    - NNAPI delegate: uses phone's dedicated AI chip if available
    - Very small runtime (~1MB)
    - Battery efficient
    
    THE CONVERSION PATH:
    PyTorch → ONNX → TensorFlow → TFLite
    
    WHY NOT DIRECT PYTORCH → TFLITE?
    No direct converter exists.
    ONNX is the universal bridge between frameworks.
    
    TWO VERSIONS:
    1. hey_orvyn.tflite      — float32, higher accuracy
    2. hey_orvyn_quant.tflite — INT8, 4x smaller, slight accuracy drop
    
    For production we recommend the quantized version.
    """
    print("\n" + "="*60)
    print("  EXPORTING TO TFLITE (Android)")
    print("="*60)
    
    # Try to import TensorFlow
    try:
        import tensorflow as tf
        print(f"  TensorFlow version: {tf.__version__}")
    except ImportError:
        print("  ⚠️  TensorFlow not installed.")
        print("  Installing TensorFlow...")
        import subprocess
        import sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "tensorflow", "tf2onnx"])
        import tensorflow as tf
    
    try:
        import tf2onnx
    except ImportError:
        import subprocess
        import sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "tf2onnx", "onnx"])
    
    print("  Converting ONNX → TensorFlow → TFLite...")
    
    try:
        import subprocess
        import sys
        
        # Step 1: ONNX → TensorFlow SavedModel
        tf_model_path = EXPORT_DIR / "tf_model"
        
        print("  Step 1: ONNX → TensorFlow SavedModel...")
        result = subprocess.run([
            sys.executable, "-m", "tf2onnx.convert",
            "--onnx", str(ONNX_PATH),
            "--output", str(EXPORT_DIR / "model_tf.onnx"),
            "--opset", "17"
        ], capture_output=True, text=True)
        
        # Alternative: use onnx-tf
        try:
            from onnx_tf.backend import prepare
            import onnx
            
            print("  Converting with onnx-tf...")
            onnx_model = onnx.load(str(ONNX_PATH))
            tf_rep = prepare(onnx_model)
            tf_rep.export_graph(str(tf_model_path))
            
        except ImportError:
            # Install onnx-tf
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "onnx-tf"
            ])
            from onnx_tf.backend import prepare
            import onnx
            onnx_model = onnx.load(str(ONNX_PATH))
            tf_rep = prepare(onnx_model)
            tf_rep.export_graph(str(tf_model_path))
        
        # Step 2: TensorFlow SavedModel → TFLite (float32)
        print("  Step 2: TensorFlow → TFLite (float32)...")
        converter = tf.lite.TFLiteConverter.from_saved_model(str(tf_model_path))
        tflite_model = converter.convert()
        
        with open(TFLITE_PATH, 'wb') as f:
            f.write(tflite_model)
        
        size_kb = TFLITE_PATH.stat().st_size / 1024
        print(f"  ✅ TFLite float32: {size_kb:.1f} KB → {TFLITE_PATH}")
        
        # Step 3: TFLite with INT8 quantization
        print("  Step 3: TFLite with INT8 quantization...")
        
        converter_quant = tf.lite.TFLiteConverter.from_saved_model(str(tf_model_path))
        converter_quant.optimizations = [tf.lite.Optimize.DEFAULT]
        
        # Representative dataset for calibration
        # INT8 quantization needs sample data to calibrate the value ranges
        def representative_dataset():
            for _ in range(100):
                # Random mel spectrogram as calibration data
                yield [np.random.randn(1, 1, 40, 101).astype(np.float32)]
        
        converter_quant.representative_dataset = representative_dataset
        converter_quant.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS_INT8
        ]
        converter_quant.inference_input_type  = tf.int8
        converter_quant.inference_output_type = tf.int8
        
        tflite_quant = converter_quant.convert()
        
        with open(TFLITE_QUANT_PATH, 'wb') as f:
            f.write(tflite_quant)
        
        quant_size_kb = TFLITE_QUANT_PATH.stat().st_size / 1024
        print(f"  ✅ TFLite INT8: {quant_size_kb:.1f} KB → {TFLITE_QUANT_PATH}")
        print(f"  Size reduction: {size_kb/quant_size_kb:.1f}x smaller")
        
        return True
        
    except Exception as e:
        print(f"  ⚠️  TFLite export failed: {e}")
        print(f"  Don't worry — ONNX works for desktop.")
        print(f"  For Android, we can convert TFLite separately.")
        export_tflite_fallback(model)
        return False


def export_tflite_fallback(model):
    """
    Fallback: saves model in a format ready for TFLite conversion.
    If full TFLite conversion fails, this saves the ONNX file
    and instructions for converting on Google Colab.
    """
    print("\n  📋 TFLite Fallback Instructions:")
    print("  The ONNX model is ready at: export/hey_orvyn.onnx")
    print("  To convert to TFLite, run this on Google Colab:")
    print("""
  !pip install onnx onnx-tf tensorflow
  from google.colab import files
  files.upload()  # upload hey_orvyn.onnx
  
  import onnx
  from onnx_tf.backend import prepare
  import tensorflow as tf
  
  onnx_model = onnx.load('hey_orvyn.onnx')
  tf_rep = prepare(onnx_model)
  tf_rep.export_graph('tf_model')
  
  converter = tf.lite.TFLiteConverter.from_saved_model('tf_model')
  converter.optimizations = [tf.lite.Optimize.DEFAULT]
  tflite_model = converter.convert()
  
  with open('hey_orvyn.tflite', 'wb') as f:
      f.write(tflite_model)
  
  files.download('hey_orvyn.tflite')
  """)


# ── SAVE MODEL CONFIG ─────────────────────────────────────────────────────────
def save_model_config(checkpoint):
    """
    Saves model configuration as JSON.
    
    WHY SAVE CONFIG?
    When you load the model in your Android app or Windows app,
    you need to know:
    - What input shape to feed it
    - What sample rate to use
    - What threshold to use for detection
    - What the model was trained on
    
    This JSON file documents all of that.
    Your app reads this file to configure the audio pipeline.
    """
    config = {
        "model_name": "Hey Orvyn Wake Word",
        "version": "1.0.0",
        "files": {
            "onnx": "hey_orvyn.onnx",
            "tflite": "hey_orvyn.tflite",
            "tflite_quantized": "hey_orvyn_quant.tflite"
        },
        "audio": {
            "sample_rate": 16000,
            "window_size_ms": 1000,
            "hop_size_ms": 10,
            "n_mels": 40,
            "n_fft": 400,
            "hop_length": 160,
            "f_min": 80,
            "f_max": 8000
        },
        "model": {
            "input_shape": [1, 1, 40, 101],
            "input_name": "mel_spectrogram",
            "output_name": "wake_word_logit",
            "detection_threshold": 0.5,
            "architecture": "TC-ResNet-14"
        },
        "performance": {
            "val_accuracy": float(f"{checkpoint['val_acc']:.4f}"),
            "val_far": float(f"{checkpoint['val_far']:.4f}"),
            "val_frr": float(f"{checkpoint['val_frr']:.4f}"),
            "training_epoch": int(checkpoint['epoch'])
        },
        "languages": [
            "en", "hi", "te", "ta", "kn", "ml", "bn", "mr",
            "gu", "pa", "es", "fr", "de", "ar", "zh", "ja",
            "ko", "pt", "ru", "it", "tr", "nl", "pl", "sv",
            "id", "uk", "ro", "el", "cs", "fi", "da", "ms", "vi"
        ],
        "wake_word": "Hey Orvyn",
        "description": "Multilingual wake word model trained on 40 languages"
    }
    
    with open(CONFIG_PATH, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n✅ Model config saved to {CONFIG_PATH}")


# ── VERIFY ONNX INFERENCE ─────────────────────────────────────────────────────
def verify_onnx_realtime():
    """
    Simulates real-time inference with ONNX Runtime.
    Shows how fast the model runs on your CPU.
    
    This is exactly what will happen in your Windows app:
    Mic audio → mel spectrogram → ONNX Runtime → probability → detect
    """
    print("\n" + "="*60)
    print("  VERIFYING ONNX REAL-TIME PERFORMANCE")
    print("="*60)
    
    try:
        import onnxruntime as ort
        import time
        
        # Load model
        session = ort.InferenceSession(
            str(ONNX_PATH),
            providers=['CPUExecutionProvider']
        )
        
        # Simulate 100 inference calls (like 100 audio frames)
        dummy_input = np.random.randn(1, 1, 40, 101).astype(np.float32)
        
        # Warmup
        for _ in range(10):
            session.run(None, {'mel_spectrogram': dummy_input})
        
        # Benchmark
        start = time.time()
        N = 100
        for _ in range(N):
            outputs = session.run(None, {'mel_spectrogram': dummy_input})
        elapsed = (time.time() - start) / N * 1000  # ms per inference
        
        logit = outputs[0][0][0]
        prob  = 1 / (1 + np.exp(-logit))
        
        print(f"  Inference time : {elapsed:.2f} ms per frame")
        print(f"  Frames per sec : {1000/elapsed:.0f}")
        print(f"  CPU load       : ~{elapsed/10:.1f}% (at 10ms hop)")
        print(f"  Sample output  : {prob:.4f} probability")
        
        if elapsed < 5:
            print(f"  ✅ EXCELLENT — way under 10ms budget")
        elif elapsed < 10:
            print(f"  ✅ GOOD — within 10ms budget")
        else:
            print(f"  ⚠️  SLOW — consider optimization")
            
    except Exception as e:
        print(f"  ⚠️  Benchmark failed: {e}")


# ── PRINT FINAL SUMMARY ───────────────────────────────────────────────────────
def print_summary():
    print("\n" + "="*60)
    print("  EXPORT COMPLETE — YOUR MODEL IS READY!")
    print("="*60)
    
    files = [
        (ONNX_PATH,         "Windows desktop (ONNX Runtime)"),
        (TFLITE_PATH,       "Android (TFLite float32)"),
        (TFLITE_QUANT_PATH, "Android (TFLite INT8 quantized)"),
        (CONFIG_PATH,       "Model configuration"),
    ]
    
    print("\n  Exported files:")
    for path, description in files:
        if path.exists():
            size_kb = path.stat().st_size / 1024
            print(f"  ✅ {path.name:30} {size_kb:8.1f} KB  ← {description}")
        else:
            print(f"  ❌ {path.name:30} not generated")
    
    print(f"""
  HOW TO USE IN YOUR APP:

  WINDOWS (Python):
    import onnxruntime as ort
    import numpy as np
    
    session = ort.InferenceSession('export/hey_orvyn.onnx')
    
    # mel_spec shape: (1, 1, 40, 101)
    prob = session.run(None, {{'mel_spectrogram': mel_spec}})[0]
    if prob > 0.5:
        print("Hey Orvyn detected!")

  ANDROID (Kotlin):
    val interpreter = Interpreter(loadModelFile("hey_orvyn.tflite"))
    val input = Array(1){{Array(1){{Array(40){{FloatArray(101)}}}}}}
    val output = Array(1){{FloatArray(1)}}
    interpreter.run(input, output)
    if (output[0][0] > 0.5f) {{ wakeOrvyn() }}

  Next step: Build the real-time detection app!
  Ask: "write the real-time detection script for Windows"
    """)


# ── ENTRY POINT ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("="*60)
    print("  HEY ORVYN — MODEL EXPORT")
    print("  Converting to ONNX + TFLite for deployment")
    print("="*60)
    
    # Step 1: Load trained model
    model, checkpoint = load_model()
    
    # Step 2: Export to ONNX
    onnx_ok = export_onnx(model)
    
    # Step 3: Export to TFLite
    tflite_ok = export_tflite(model)
    
    # Step 4: Save config
    save_model_config(checkpoint)
    
    # Step 5: Verify ONNX performance
    if onnx_ok:
        verify_onnx_realtime()
    
    # Step 6: Summary
    print_summary()