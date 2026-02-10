import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from model_mlm import TransformerMLM
from tokenizer_bpe import BPETokenizer
import pandas as pd
import numpy as np
from tqdm import tqdm
import random
from torch.optim.lr_scheduler import LambdaLR
import math

# Configuration
BATCH_SIZE = 64
EPOCHS = 50  # Increased for better semantic understanding
LEARNING_RATE = 5e-5
MAX_LEN = 128
D_MODEL = 512
NUM_HEADS = 8
NUM_LAYERS = 4
D_FF = 2048  # 4x D_MODEL
DATA_FILE = 'online_shopping_10_cats.csv'
VOCAB_FILE = 'vocab_bpe.json'
MASK_PROB = 0.15
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

class MLMDataset(Dataset):
    """Dataset for Masked Language Modeling"""
    def __init__(self, texts, tokenizer, max_len=128, mask_prob=0.15):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.mask_prob = mask_prob
        self.mask_token_id = tokenizer.unk_token_id  # Use UNK as MASK
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        
        # Encode
        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        
        # Create labels (same as input_ids)
        labels = input_ids.clone()
        
        # Mask random tokens
        probability_matrix = torch.rand(input_ids.shape)
        
        # Don't mask PAD tokens
        special_tokens_mask = (input_ids == self.tokenizer.pad_token_id)
        probability_matrix.masked_fill_(special_tokens_mask, 0.0)
        
        # Mask tokens with MASK_PROB probability
        masked_indices = probability_matrix < self.mask_prob
        
        # Set labels to -100 for non-masked tokens (ignore in loss)
        labels[~masked_indices] = -100
        
        # Replace masked tokens with MASK token
        input_ids[masked_indices] = self.mask_token_id
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

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
    total_predictions = 0
    
    loop = tqdm(data_loader, desc=f'Pre-train Epoch {epoch + 1}/{EPOCHS}')
    
    for d in loop:
        input_ids = d['input_ids'].to(device)
        attention_mask = d['attention_mask'].to(device)
        labels = d['labels'].to(device)
        
        # Forward pass: [batch_size, seq_len + 1, vocab_size]
        logits = model(input_ids, attention_mask)
        
        # Adjust labels: Prepend -100 for the [CLS] token (index 0)
        # logits has 1 extra position at the start for [CLS]
        batch_size = labels.size(0)
        cls_labels = torch.full((batch_size, 1), -100).to(device)
        full_labels = torch.cat((cls_labels, labels), dim=1) # [batch, seq+1]
        
        # Reshape for loss calculation
        loss = criterion(logits.view(-1, logits.size(-1)), full_labels.view(-1))
        
        # Calculate accuracy on masked tokens
        mask = (full_labels != -100)
        if mask.sum() > 0:
            preds = torch.argmax(logits, dim=-1)
            correct_predictions += ((preds == full_labels) & mask).sum().item()
            total_predictions += mask.sum().item()
        
        losses.append(loss.item())
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        loop.set_postfix(loss=loss.item(), lr=optimizer.param_groups[0]['lr'])
    
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    return accuracy, np.mean(losses)

def main():
    print(f"Using device: {DEVICE}")
    
    # Load data
    df = pd.read_csv(DATA_FILE)
    texts = df['review'].astype(str).tolist()
    print(f"Loaded {len(texts)} reviews")

    # Step 1: Retrain Tokenizer with fixed BPE logic
    print("Retraining BPE Tokenizer for higher quality subwords...")
    tokenizer = BPETokenizer(vocab_size=5000)
    tokenizer.fit_on_texts(texts)
    tokenizer.save_vocab(VOCAB_FILE)
    
    vocab_size = tokenizer.vocab_size_actual
    print(f"Vocab size: {vocab_size}")
    
    # Create dataset
    dataset = MLMDataset(texts, tokenizer, MAX_LEN, MASK_PROB)
    # Optimization: Use num_workers and pin_memory to prevent data loading bottleneck
    train_loader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    # Initialize MLM model
    print("Initializing TransformerMLM for extended pre-training...")
    model = TransformerMLM(
        vocab_size=vocab_size,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        d_ff=D_FF
    )
    model = model.to(DEVICE)
    
    # Loss ignores -100 labels to avoid calculating gradients for unmasked positions
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    
    # Using AdamW for better decoupling of weight decay and gradients
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    
    # Cosine scheduler with warmup helps stabilize deep Transformer training in early steps
    total_steps = len(train_loader) * EPOCHS
    warmup_steps = int(total_steps * 0.1) 
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    
    best_loss = float('inf')
    
    for epoch in range(EPOCHS):
        # MLM training epoch
        acc, loss = train_epoch(model, train_loader, criterion, optimizer, scheduler, DEVICE, epoch)
        
        print(f'Epoch {epoch+1}/{EPOCHS} - Loss: {loss:.4f}, MLM Accuracy: {acc:.4f}')
        
        # Save the best encoder for downstream classification tasks
        if loss < best_loss:
            best_loss = loss
            model.save_encoder('pretrained_encoder.bin')
            print(f"New best loss: {best_loss:.4f}. Encoder checkpoint updated.")
    
    print("Pre-training phase complete.")

if __name__ == "__main__":
    main()
