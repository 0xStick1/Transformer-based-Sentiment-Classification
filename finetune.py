import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from dataset import WaimaiDataset
from model_scratch import SimpleTransformer
from model_mlm import TransformerMLM
from tokenizer_bpe import BPETokenizer
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
from torch.optim.lr_scheduler import LambdaLR
import math

from torch.optim.swa_utils import AveragedModel, SWALR
import random

# Configuration
BATCH_SIZE = 64
EPOCHS = 60 # Marathon fine-tuning
LEARNING_RATE_ENCODER = 1e-5
LEARNING_RATE_HEAD = 1e-4
MAX_LEN = 128
D_MODEL = 512
NUM_HEADS = 8
NUM_LAYERS = 4
D_FF = 2048
DATA_FILE = 'merged_sentiment_data.csv'
VOCAB_FILE = 'vocab_bpe.json'
PRETRAINED_ENCODER = 'pretrained_encoder.bin'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda)

def train_epoch(model, data_loader, criterion, optimizer, scheduler, device, epoch):
    model.train()
    losses = []
    correct_predictions = 0
    
    loop = tqdm(data_loader, desc=f'Fine-tune Epoch {epoch + 1}/{EPOCHS}')
    
    for d in loop:
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        labels = d["labels"].to(device)
        
        logits = model(input_ids, attention_mask=attention_mask)
        loss = criterion(logits, labels)
        
        preds = torch.argmax(logits, dim=1)
        correct_predictions += torch.sum(preds == labels)
        losses.append(loss.item())
        
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        loop.set_postfix(loss=loss.item(), lr=optimizer.param_groups[0]['lr'])
        
    return (correct_predictions.double() / len(data_loader.dataset)).item(), np.mean(losses)

def eval_model(model, data_loader, criterion, device):
    model.eval()
    losses = []
    correct_predictions = 0
    
    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            labels = d["labels"].to(device)
            
            logits = model(input_ids, attention_mask=attention_mask)
            loss = criterion(logits, labels)
            
            preds = torch.argmax(logits, dim=1)
            correct_predictions += torch.sum(preds == labels)
            losses.append(loss.item())
            
    return (correct_predictions.double() / len(data_loader.dataset)).item(), np.mean(losses)

def main():
    print(f"Using device: {DEVICE}")
    
    # Load tokenizer
    print("Loading BPE Tokenizer...")
    tokenizer = BPETokenizer.load_vocab(VOCAB_FILE)
    vocab_size = tokenizer.vocab_size_actual
    print(f"Vocab size: {vocab_size}")
    
    # Prepare dataset
    dataset = WaimaiDataset(DATA_FILE, tokenizer, MAX_LEN, augment=True)  # Enable Augmentation for 95% Target
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Disable augmentation for validation
    val_dataset.dataset.augment = False 
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Initialize model
    print("Initializing SimpleTransformer for classification...")
    model = SimpleTransformer(
        vocab_size=vocab_size,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        d_ff=D_FF,  # Align with pre-training
        num_classes=2
    )
    
    # Load pre-trained encoder weights
    if os.path.exists(PRETRAINED_ENCODER):
        print(f"Loading pre-trained encoder from {PRETRAINED_ENCODER}...")
        model = TransformerMLM.load_encoder_to_classifier(PRETRAINED_ENCODER, model)
        print("Pre-trained encoder loaded successfully!")
    else:
        print(f"Warning: {PRETRAINED_ENCODER} not found. Training from scratch.")
        print("Run 'python pretrain.py' first for better results.")
    
    model = model.to(DEVICE)
    
    # 1. Differential Learning Rates
    # Experiments show that using a lower LR (1e-5) for the pre-trained encoder 
    # protects extracted features, while a higher LR (1e-4) for the classifier 
    # head accelerates convergence on the specific sentiment task.
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if 'classifier' not in n], 'lr': LEARNING_RATE_ENCODER},
        {'params': [p for n, p in model.named_parameters() if 'classifier' in n], 'lr': LEARNING_RATE_HEAD}
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters)
    
    # 2. SWA (Stochastic Weight Averaging)
    # SWA averages weights over the final training trajectory, which provides 
    # significant gains in generalization and stabilizes validation performance.
    swa_model = AveragedModel(model)
    swa_start = 40 # Trigger SWA averaging after initial convergence on 193k data
    swa_scheduler = SWALR(optimizer, swa_lr=LEARNING_RATE_ENCODER)
    
    # Cosine scheduler for the pre-SWA phase to maintain smooth descent
    total_steps = len(train_loader) * swa_start
    warmup_steps = int(total_steps * 0.1)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    
    criterion = nn.CrossEntropyLoss()
    best_accuracy = 0

    for epoch in range(EPOCHS):
        val_acc = 0
        if epoch < swa_start:
            # Standard Fine-tuning phase
            train_acc, train_loss = train_epoch(model, train_loader, criterion, optimizer, scheduler, DEVICE, epoch)
            print(f'Train loss {train_loss:.4f} accuracy {train_acc:.4f}')
            val_acc, val_loss = eval_model(model, val_loader, criterion, DEVICE)
            print(f'Val   loss {val_loss:.4f} accuracy {val_acc:.4f}')
        else:
            # SWA phase: weights are updated in every step and averaged in every epoch
            print(f"--- SWA Trajectory Averaging (Epoch {epoch+1}) ---")
            train_acc, train_loss = train_epoch(model, train_loader, criterion, optimizer, swa_scheduler, DEVICE, epoch)
            print(f'Train loss {train_loss:.4f} accuracy {train_acc:.4f}')
            swa_model.update_parameters(model)
            
            # Evaluate using the averaged model
            val_acc, val_loss = eval_model(swa_model, val_loader, criterion, DEVICE)
            print(f'SWA Val accuracy {val_acc:.4f}')
            swa_scheduler.step()

        # Checkpointing logic
        if val_acc > best_accuracy:
            best_accuracy = val_acc
            # Save SWA model if in SWA phase, otherwise base model
            save_model = swa_model if epoch >= swa_start else model
            torch.save(save_model.state_dict(), 'best_model_finetuned.bin')
            print(f"New best performance recorded: {best_accuracy:.4f}")

    print(f"Fine-tuning session ended. Final Best Accuracy: {best_accuracy:.4f}")

if __name__ == "__main__":
    main()
