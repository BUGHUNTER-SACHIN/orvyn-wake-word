"""
=============================================================
  Hey Orvyn - Wake Word Model Training Script
  Trains a TC-ResNet model on your RTX 4060 GPU
=============================================================

HOW TO RUN:
  python train.py

WHAT IT DOES:
  1. Loads your 18,709 processed mel spectrograms
  2. Splits into train (80%) and validation (20%)
  3. Trains a TC-ResNet-14 neural network
  4. Evaluates FAR and FRR after every epoch
  5. Saves the best model checkpoint
  6. Plots training curves when done

EXPECTED RESULTS ON RTX 4060:
  Training time : 5-10 minutes
  Final accuracy: >95%
  FAR target    : <5%
  FRR target    : <8%

WHAT YOU'LL LEARN:
  - How a training loop works
  - What loss, accuracy, FAR, FRR mean
  - How to read training curves
  - Why we use GPU for training
"""

# ── IMPORTS ───────────────────────────────────────────────────────────────────
import os
import csv
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# ── CONFIGURATION ─────────────────────────────────────────────────────────────
#
# These are all the settings that control how training works.
# Grouped here so you can find and change them easily.
#
# BATCH_SIZE = 128
#   How many samples we show the model at once.
#   Larger = faster training but needs more GPU memory.
#   128 is safe for RTX 4060 (8GB VRAM).
#
# EPOCHS = 50
#   How many times we go through the entire dataset.
#   Each epoch the model sees all 18,709 samples.
#   50 epochs is usually enough for wake word models.
#
# LEARNING_RATE = 0.001
#   How big each adjustment step is during training.
#   Too high = model overshoots and never converges.
#   Too low  = model learns very slowly.
#   0.001 is the standard starting point.
#
# POS_WEIGHT = 9.0
#   Our dataset has 1,709 positives and 17,000 negatives.
#   That's a 1:9 ratio — imbalanced!
#   Without weighting, the model would just predict "negative"
#   for everything and be 90% accurate but useless.
#   pos_weight=9 tells the model: "mistakes on positive samples
#   are 9x more costly than mistakes on negatives."
#   This forces the model to actually learn "Hey Orvyn".

METADATA_FILE  = Path("data/processed/metadata.csv")
MODELS_DIR     = Path("models")
BATCH_SIZE     = 128
EPOCHS         = 50
LEARNING_RATE  = 0.001
POS_WEIGHT     = 9.0
VAL_SPLIT      = 0.2    # 20% of data for validation
DETECTION_THRESHOLD = 0.5  # above this = wake word detected

# Use GPU if available (your RTX 4060)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── DATASET CLASS ─────────────────────────────────────────────────────────────
#
# WHAT IS A DATASET CLASS?
# PyTorch needs a special class to load your data.
# It works like a list — PyTorch asks "give me sample #542"
# and the Dataset class loads that .npy file and returns it.
#
# WHY NOT LOAD ALL DATA AT ONCE?
# 18,709 samples × (40×101) floats = ~300MB.
# Loading all at once is fine for this size, but for larger
# datasets you'd load one at a time to save memory.
# We cache everything in memory here since 300MB fits easily.

class WakeWordDataset(Dataset):
    """
    Custom PyTorch Dataset for Hey Orvyn wake word data.
    
    Loads mel spectrograms from .npy files and their labels
    from metadata.csv. Returns (spectrogram, label) pairs.
    """
    
    def __init__(self, metadata_file, indices=None):
        """
        metadata_file : path to metadata.csv
        indices       : which rows to use (for train/val split)
        """
        self.samples = []
        
        # Read metadata.csv
        with open(metadata_file, 'r') as f:
            reader = csv.DictReader(f)
            all_rows = list(reader)
        
        # Use only specified indices (for train/val split)
        if indices is not None:
            all_rows = [all_rows[i] for i in indices]
        
        print(f"  Loading {len(all_rows)} samples into memory...")
        
        # Load all .npy files into memory
        # This makes training fast — no disk access during training
        loaded = 0
        skipped = 0
        
        for row in tqdm(all_rows, desc="  Loading", unit="file"):
            filepath = row['filepath']
            label    = int(row['label'])
            
            if not Path(filepath).exists():
                skipped += 1
                continue
            
            try:
                # Load mel spectrogram
                mel_spec = np.load(filepath).astype(np.float32)
                
                # Verify shape is correct (40, 101)
                if mel_spec.shape != (40, 101):
                    skipped += 1
                    continue
                
                self.samples.append((mel_spec, label))
                loaded += 1
                
            except Exception:
                skipped += 1
        
        print(f"  Loaded: {loaded}, Skipped: {skipped}")
        
        # Count positives and negatives
        pos = sum(1 for _, l in self.samples if l == 1)
        neg = sum(1 for _, l in self.samples if l == 0)
        print(f"  Positives: {pos}, Negatives: {neg}")
    
    def __len__(self):
        """Returns total number of samples."""
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Returns one sample as (tensor, label).
        
        PyTorch calls this automatically during training.
        It asks for sample #0, #1, #2... in random order.
        
        We add a channel dimension: (40, 101) → (1, 40, 101)
        WHY: CNNs expect (channels, height, width).
        Like RGB images have 3 channels, our spectrogram has 1.
        """
        mel_spec, label = self.samples[idx]
        
        # Add channel dimension: (40, 101) → (1, 40, 101)
        tensor = torch.tensor(mel_spec).unsqueeze(0)
        
        return tensor, torch.tensor(label, dtype=torch.float32)


# ── TC-RESNET MODEL ───────────────────────────────────────────────────────────
#
# WHAT IS TC-RESNET?
# TC = Temporal Convolution
# ResNet = Residual Network
#
# TEMPORAL CONVOLUTION:
# Instead of looking at 2D patches (like image CNNs),
# temporal convolutions look along the TIME axis of the spectrogram.
# This makes sense for audio — patterns happen over time.
# "Hey" comes before "Orvyn" — the time order matters.
#
# RESIDUAL CONNECTION:
# Instead of: output = layer(input)
# ResNet does: output = layer(input) + input
#
# WHY ADD THE INPUT BACK?
# It creates a "shortcut" — the gradient can flow directly
# through the shortcut during backpropagation.
# This prevents the "vanishing gradient" problem in deep networks.
# ResNet was invented by Microsoft in 2015 and revolutionized deep learning.
#
# OUR MODEL ARCHITECTURE:
# Input: (batch, 1, 40, 101)  ← batch of mel spectrograms
#   ↓ Conv block 1
#   ↓ Residual block 1
#   ↓ Residual block 2  
#   ↓ Residual block 3
#   ↓ Global Average Pooling  ← collapses spatial dimensions
#   ↓ Linear layer
# Output: (batch, 1)  ← one probability per sample

class ResidualBlock(nn.Module):
    """
    One residual block: two conv layers with a skip connection.
    
    WHAT HAPPENS INSIDE:
    input → Conv → BatchNorm → ReLU → Conv → BatchNorm → + input → ReLU
                                                           ↑
                                                    skip connection
    
    BATCHNORM: normalizes activations to have mean=0, std=1
    WHY: prevents values from getting too large/small during training
    
    RELU: Rectified Linear Unit — max(0, x)
    WHY: adds non-linearity. Without it, the whole network
    would just be one big matrix multiplication (linear).
    Non-linearity lets the network learn complex patterns.
    """
    
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        
        # Main path: two convolutions
        self.conv1 = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=(3, 1),    # 3×1 kernel = temporal convolution
            stride=(stride, 1),    # stride along time axis
            padding=(1, 0),        # same padding to preserve size
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=(3, 1),
            stride=1,
            padding=(1, 0),
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.relu = nn.ReLU(inplace=True)
        
        # Skip connection
        # If channels or size changes, we need to adjust the skip
        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels,
                    kernel_size=1,
                    stride=(stride, 1),
                    bias=False
                ),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        """
        Forward pass: main path + skip connection.
        
        WHAT IS FORWARD PASS?
        When we feed data through the network, it flows
        through each layer's forward() method in order.
        PyTorch calls this automatically.
        """
        # Main path
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        # Add skip connection
        out = out + self.skip(x)
        out = self.relu(out)
        
        return out


class TCResNet(nn.Module):
    """
    TC-ResNet-14 for wake word detection.
    
    ARCHITECTURE:
    - Initial conv: 1 → 16 channels
    - Block 1: 16 → 24 channels (stride 2 = downsample)
    - Block 2: 24 → 32 channels (stride 2 = downsample)  
    - Block 3: 32 → 48 channels (stride 2 = downsample)
    - Global Average Pool: collapses to (batch, 48, 1, 1)
    - Linear: 48 → 1 (final prediction)
    
    TOTAL PARAMETERS: ~45,000 (very small — runs on any phone)
    """
    
    def __init__(self):
        super().__init__()
        
        # Initial convolution
        # (batch, 1, 40, 101) → (batch, 16, 40, 101)
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(3, 1), padding=(1, 0), bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        
        # Residual blocks — each doubles channels, halves time dimension
        self.block1 = ResidualBlock(16, 24, stride=2)   # time: 101 → 51
        self.block2 = ResidualBlock(24, 32, stride=2)   # time: 51 → 26
        self.block3 = ResidualBlock(32, 48, stride=2)   # time: 26 → 13
        
        # Global Average Pooling
        # Averages across ALL spatial positions
        # (batch, 48, 5, 13) → (batch, 48, 1, 1)
        # WHY: makes the model work on any input size
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Dropout for regularization
        # WHY: randomly zeros 30% of neurons during training
        # This prevents overfitting — model can't rely on any one neuron
        self.dropout = nn.Dropout(0.3)
        
        # Final linear layer: 48 features → 1 probability
        self.fc = nn.Linear(48, 1)
    
    def forward(self, x):
        """
        Full forward pass through the network.
        
        Input:  (batch, 1, 40, 101)
        Output: (batch, 1) — raw logit (before sigmoid)
        """
        x = self.conv1(x)    # → (batch, 16, 40, 101)
        x = self.block1(x)   # → (batch, 24, 20, 101)
        x = self.block2(x)   # → (batch, 32, 10, 101)
        x = self.block3(x)   # → (batch, 48, 5, 101)
        x = self.gap(x)      # → (batch, 48, 1, 1)
        x = x.view(x.size(0), -1)  # flatten → (batch, 48)
        x = self.dropout(x)
        x = self.fc(x)       # → (batch, 1)
        return x
    
    def count_parameters(self):
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── METRICS: FAR AND FRR ──────────────────────────────────────────────────────
#
# FAR = False Accept Rate
#   How often the model says "wake word!" when it's NOT "Hey Orvyn"
#   FAR = false_positives / total_negatives
#   TARGET: < 5% (ideally < 1%)
#   HIGH FAR = model triggers randomly — very annoying for users
#
# FRR = False Reject Rate  
#   How often the model misses "Hey Orvyn" when you actually said it
#   FRR = false_negatives / total_positives
#   TARGET: < 8%
#   HIGH FRR = model doesn't respond when you speak — frustrating
#
# THE TRADEOFF:
# Lowering threshold → FAR goes up, FRR goes down (more sensitive)
# Raising threshold → FAR goes down, FRR goes up (less sensitive)
# We pick threshold=0.5 as starting point, tune later.

def calculate_metrics(predictions, labels, threshold=DETECTION_THRESHOLD):
    """
    Calculates accuracy, FAR, and FRR.
    
    predictions : raw model outputs (logits)
    labels      : true labels (0 or 1)
    threshold   : above this = wake word detected
    """
    # Apply sigmoid to convert logits to probabilities (0 to 1)
    # logit=0 → probability=0.5
    # logit=2 → probability=0.88
    # logit=-2 → probability=0.12
    probs = torch.sigmoid(predictions).cpu().numpy().flatten()
    labels_np = labels.cpu().numpy().flatten()
    
    # Apply threshold
    predicted = (probs >= threshold).astype(int)
    
    # Calculate confusion matrix
    # [[TN, FP],
    #  [FN, TP]]
    cm = confusion_matrix(labels_np, predicted, labels=[0, 1])
    
    tn = cm[0, 0]  # True Negative  (correctly rejected non-wake-word)
    fp = cm[0, 1]  # False Positive (wrongly accepted non-wake-word) → FAR
    fn = cm[1, 0]  # False Negative (wrongly rejected wake-word) → FRR
    tp = cm[1, 1]  # True Positive  (correctly detected wake-word)
    
    # Calculate rates
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    far = fp / (fp + tn + 1e-8)  # false accepts / total negatives
    frr = fn / (fn + tp + 1e-8)  # false rejects / total positives
    
    return accuracy, far, frr


# ── TRAINING LOOP ─────────────────────────────────────────────────────────────
#
# THE TRAINING LOOP — most important concept in deep learning:
#
# For each epoch:
#   For each batch:
#     1. FORWARD PASS: feed batch through network → get predictions
#     2. LOSS: measure how wrong predictions are
#     3. BACKWARD PASS: calculate gradients (which direction to adjust)
#     4. OPTIMIZER STEP: adjust weights in that direction
#
# This 4-step loop runs 18,709/128 = 146 times per epoch.
# After 50 epochs = 7,300 total updates.
# Each update makes the model slightly better at detecting "Hey Orvyn".

def train_epoch(model, loader, criterion, optimizer, device):
    """
    Runs one complete pass through the training data.
    Returns average loss for this epoch.
    """
    model.train()  # tells model we're training (enables dropout, batchnorm updates)
    
    total_loss = 0
    all_preds  = []
    all_labels = []
    
    for batch_spectrograms, batch_labels in loader:
        # Move data to GPU
        # This is why GPU training is fast — data lives on GPU memory
        batch_spectrograms = batch_spectrograms.to(device)
        batch_labels       = batch_labels.to(device).unsqueeze(1)
        
        # Step 1: Zero gradients from previous batch
        # WHY: gradients accumulate by default in PyTorch
        # We clear them each batch so previous batch doesn't affect this one
        optimizer.zero_grad()
        
        # Step 2: Forward pass — feed data through network
        predictions = model(batch_spectrograms)
        
        # Step 3: Calculate loss
        # BCEWithLogitsLoss = Binary Cross Entropy + Sigmoid combined
        # It measures: how different are predictions from true labels?
        # Loss = 0 means perfect predictions
        # Loss = 1+ means bad predictions
        loss = criterion(predictions, batch_labels)
        
        # Step 4: Backward pass — calculate gradients
        # PyTorch automatically calculates how much each weight
        # contributed to the loss (chain rule / backpropagation)
        loss.backward()
        
        # Step 5: Gradient clipping
        # Prevents gradients from getting too large (exploding gradients)
        # Clips gradient norm to max value of 1.0
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        # Step 6: Optimizer step — update weights
        # AdamW adjusts each weight based on its gradient
        # Weights that caused more error get adjusted more
        optimizer.step()
        
        total_loss += loss.item()
        all_preds.append(predictions.detach())
        all_labels.append(batch_labels.detach())
    
    # Calculate metrics for this epoch
    all_preds  = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    acc, far, frr = calculate_metrics(all_preds, all_labels)
    avg_loss = total_loss / len(loader)
    
    return avg_loss, acc, far, frr


def validate_epoch(model, loader, criterion, device):
    """
    Evaluates model on validation data (data it hasn't trained on).
    
    WHY VALIDATE ON SEPARATE DATA?
    If we only measure performance on training data, the model
    might just memorize the answers without learning real patterns.
    Validation data tells us how well it generalizes to new audio.
    
    model.eval() disables dropout and freezes batchnorm.
    torch.no_grad() disables gradient calculation (saves memory/time).
    """
    model.eval()
    
    total_loss = 0
    all_preds  = []
    all_labels = []
    
    with torch.no_grad():
        for batch_spectrograms, batch_labels in loader:
            batch_spectrograms = batch_spectrograms.to(device)
            batch_labels       = batch_labels.to(device).unsqueeze(1)
            
            predictions = model(batch_spectrograms)
            loss        = criterion(predictions, batch_labels)
            
            total_loss += loss.item()
            all_preds.append(predictions)
            all_labels.append(batch_labels)
    
    all_preds  = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    acc, far, frr = calculate_metrics(all_preds, all_labels)
    avg_loss = total_loss / len(loader)
    
    return avg_loss, acc, far, frr


# ── PLOT TRAINING CURVES ──────────────────────────────────────────────────────
def plot_training_curves(history, save_path):
    """
    Plots loss, accuracy, FAR, FRR over all epochs.
    
    WHAT TO LOOK FOR:
    - Loss should go DOWN over epochs (model is learning)
    - Accuracy should go UP
    - FAR should go DOWN (fewer false triggers)
    - FRR should go DOWN (fewer missed wake words)
    
    If training loss goes down but val loss goes UP = overfitting
    This means model memorized training data but can't generalize.
    Solution: more data, more dropout, or fewer epochs.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Hey Orvyn — Training Curves', fontsize=14)
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss
    axes[0,0].plot(epochs, history['train_loss'], 'b-', label='Train')
    axes[0,0].plot(epochs, history['val_loss'],   'r-', label='Val')
    axes[0,0].set_title('Loss (lower = better)')
    axes[0,0].set_xlabel('Epoch')
    axes[0,0].legend()
    axes[0,0].grid(True)
    
    # Accuracy
    axes[0,1].plot(epochs, history['train_acc'], 'b-', label='Train')
    axes[0,1].plot(epochs, history['val_acc'],   'r-', label='Val')
    axes[0,1].set_title('Accuracy (higher = better)')
    axes[0,1].set_xlabel('Epoch')
    axes[0,1].set_ylim([0, 1])
    axes[0,1].legend()
    axes[0,1].grid(True)
    
    # FAR
    axes[1,0].plot(epochs, history['val_far'], 'r-')
    axes[1,0].axhline(y=0.05, color='g', linestyle='--', label='Target (5%)')
    axes[1,0].set_title('False Accept Rate (lower = better)')
    axes[1,0].set_xlabel('Epoch')
    axes[1,0].legend()
    axes[1,0].grid(True)
    
    # FRR
    axes[1,1].plot(epochs, history['val_frr'], 'b-')
    axes[1,1].axhline(y=0.08, color='g', linestyle='--', label='Target (8%)')
    axes[1,1].set_title('False Reject Rate (lower = better)')
    axes[1,1].set_xlabel('Epoch')
    axes[1,1].legend()
    axes[1,1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"\n✅ Training curves saved to {save_path}")


# ── MAIN TRAINING FUNCTION ────────────────────────────────────────────────────
def train():
    print("="*60)
    print("  HEY ORVYN — MODEL TRAINING")
    print(f"  Device: {DEVICE}")
    if DEVICE.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print("="*60)
    
    # ── LOAD DATASET ──────────────────────────────────────────
    print("\n📂 Loading dataset...")
    
    # Read all metadata
    with open(METADATA_FILE, 'r') as f:
        all_rows = list(csv.DictReader(f))
    
    total = len(all_rows)
    print(f"  Total samples: {total}")
    
    # Shuffle and split into train/val
    # WHY SHUFFLE? So train and val sets have similar distributions.
    # If we don't shuffle, val might have only one language.
    indices = list(range(total))
    np.random.seed(42)  # seed=42 means same shuffle every run (reproducible)
    np.random.shuffle(indices)
    
    split = int(total * (1 - VAL_SPLIT))
    train_indices = indices[:split]
    val_indices   = indices[split:]
    
    print(f"  Train: {len(train_indices)} samples")
    print(f"  Val:   {len(val_indices)} samples")
    
    # Create datasets
    print("\n  Loading training data:")
    train_dataset = WakeWordDataset(METADATA_FILE, train_indices)
    print("\n  Loading validation data:")
    val_dataset   = WakeWordDataset(METADATA_FILE, val_indices)
    
    # Create DataLoaders
    # DataLoader handles batching, shuffling, and parallel loading
    # num_workers=0 on Windows (multiprocessing issues with >0)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,      # shuffle every epoch
        num_workers=0,
        pin_memory=True    # faster GPU transfer
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    # ── CREATE MODEL ──────────────────────────────────────────
    print("\n🧠 Creating TC-ResNet model...")
    model = TCResNet().to(DEVICE)
    params = model.count_parameters()
    print(f"  Parameters: {params:,} ({params/1000:.1f}K)")
    print(f"  Model size: ~{params * 4 / 1024:.1f} KB (float32)")
    
    # ── LOSS FUNCTION ─────────────────────────────────────────
    #
    # BCEWithLogitsLoss = Binary Cross Entropy with Logits
    # 
    # WHAT IS BINARY CROSS ENTROPY?
    # Measures how different our prediction is from the true label.
    # Perfect prediction → loss = 0
    # Completely wrong  → loss = very large number
    #
    # pos_weight = 9.0 because we have 9x more negatives than positives.
    # This balances the dataset mathematically without resampling.
    
    pos_weight = torch.tensor([POS_WEIGHT]).to(DEVICE)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # ── OPTIMIZER ─────────────────────────────────────────────
    #
    # AdamW = Adam with Weight Decay
    # Adam is the most popular optimizer for deep learning.
    #
    # WHAT IS AN OPTIMIZER?
    # After backprop calculates gradients (which direction to move),
    # the optimizer decides HOW MUCH to move in that direction.
    #
    # AdamW improvements over basic gradient descent:
    # - Adaptive learning rates per parameter
    # - Momentum: remembers past gradients to smooth updates
    # - Weight decay: prevents weights from growing too large
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4  # L2 regularization
    )
    
    # ── LEARNING RATE SCHEDULER ───────────────────────────────
    #
    # CosineAnnealingLR reduces learning rate over time.
    # Starts at 0.001, smoothly decreases to near 0 by epoch 50.
    #
    # WHY REDUCE LEARNING RATE?
    # Early training: big steps are fine (far from optimum)
    # Late training: small steps needed (close to optimum, don't overshoot)
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS,
        eta_min=1e-6
    )
    
    # ── TRAINING LOOP ─────────────────────────────────────────
    print(f"\n🚀 Training for {EPOCHS} epochs on {DEVICE}...")
    print("  Watch: loss↓  accuracy↑  FAR↓  FRR↓")
    print("="*60)
    
    MODELS_DIR.mkdir(exist_ok=True)
    
    best_val_loss = float('inf')
    best_epoch    = 0
    
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc':  [], 'val_acc':  [],
        'val_far':    [], 'val_frr':  []
    }
    
    start_time = time.time()
    
    for epoch in range(1, EPOCHS + 1):
        
        # Train one epoch
        train_loss, train_acc, train_far, train_frr = train_epoch(
            model, train_loader, criterion, optimizer, DEVICE
        )
        
        # Validate
        val_loss, val_acc, val_far, val_frr = validate_epoch(
            model, val_loader, criterion, DEVICE
        )
        
        # Step scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['val_far'].append(val_far)
        history['val_frr'].append(val_frr)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch    = epoch
            torch.save({
                'epoch':      epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss':   val_loss,
                'val_acc':    val_acc,
                'val_far':    val_far,
                'val_frr':    val_frr,
            }, MODELS_DIR / 'best_model.pt')
            saved = "✅ SAVED"
        else:
            saved = ""
        
        # Print progress every epoch
        elapsed = time.time() - start_time
        print(
            f"  Epoch {epoch:3d}/{EPOCHS} | "
            f"Loss: {train_loss:.4f}/{val_loss:.4f} | "
            f"Acc: {val_acc:.3f} | "
            f"FAR: {val_far:.3f} | "
            f"FRR: {val_frr:.3f} | "
            f"LR: {current_lr:.6f} {saved}"
        )
    
    total_time = time.time() - start_time
    
    # ── FINAL SUMMARY ─────────────────────────────────────────
    print("\n" + "="*60)
    print("  TRAINING COMPLETE!")
    print("="*60)
    print(f"  Total time    : {total_time/60:.1f} minutes")
    print(f"  Best epoch    : {best_epoch}")
    print(f"  Best val loss : {best_val_loss:.4f}")
    
    # Load best model and show final metrics
    checkpoint = torch.load(MODELS_DIR / 'best_model.pt', weights_only=False)
    print(f"  Best val acc  : {checkpoint['val_acc']:.4f} ({checkpoint['val_acc']*100:.1f}%)")
    print(f"  Best val FAR  : {checkpoint['val_far']:.4f} ({checkpoint['val_far']*100:.1f}%)")
    print(f"  Best val FRR  : {checkpoint['val_frr']:.4f} ({checkpoint['val_frr']*100:.1f}%)")
    
    # Check if we hit targets
    far_ok = checkpoint['val_far'] < 0.05
    frr_ok = checkpoint['val_frr'] < 0.08
    
    print(f"\n  FAR < 5%  : {'✅ YES' if far_ok else '❌ NO — need more training'}")
    print(f"  FRR < 8%  : {'✅ YES' if frr_ok else '❌ NO — need more training'}")
    
    # Plot curves
    plot_training_curves(history, MODELS_DIR / 'training_curves.png')
    
    print(f"""
  ✅ Model saved to: models/best_model.pt
  ✅ Curves saved to: models/training_curves.png
  
  Next step: Export model to ONNX (desktop) and TFLite (Android)
  Ask: "write the export script"
    """)


# ── ENTRY POINT ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    train()