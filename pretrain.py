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
import re

# Configuration
BATCH_SIZE = 64
EPOCHS = 100  # Marathon pre-training for deep semantic understanding
LEARNING_RATE = 5e-5
MAX_LEN = 128
D_MODEL = 512
NUM_HEADS = 8
NUM_LAYERS = 4
D_FF = 2048  # 4x D_MODEL
DATA_FILE = 'merged_sentiment_data.csv'
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
    """
    Dataset for Masked Language Modeling with Whole Word Masking (WWM).
    WWM is particularly effective for Chinese, as it prevents the model from 
    guessing a missing character simply by looking at its immediate neighbor 
    within the same word.
    """
    def __init__(self, texts, tokenizer, max_len=128, mask_prob=0.15):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.mask_prob = mask_prob
        self.mask_token_id = tokenizer.unk_token_id  # Use UNK as MASK for this scratch implementation
        
    def _get_word_groups(self, text):
        """
        Heuristic-based Chinese word segmentation for WWM.
        Groups contiguous Chinese characters or alphanumeric sequences.
        """
        # Regex to segment text into word-like chunks (Chinese spans, alphanumeric spans, or punctuation)
        words = re.findall(r'[\u4e00-\u9fa5]+|[a-zA-Z0-9]+|[^\u4e00-\u9fa5a-zA-Z0-9]', text)
        
        token_groups = []
        for word in words:
            # Tokenize each word chunk separately using the BPE merges
            word_tokens = self.tokenizer._apply_merges(word)
            word_ids = [self.tokenizer.vocab.get(t, self.tokenizer.unk_token_id) for t in word_tokens]
            if word_ids:
                token_groups.append(word_ids)
        return token_groups

    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx]).strip()
        
        # 1. Segment text into word groups to support WWM
        word_groups = self._get_word_groups(text)
        
        # 2. Flatten groups into a single sequence while tracking grouping
        input_ids = []
        group_map = [] # Tracks which tokens belong to which word group
        
        for group_idx, group in enumerate(word_groups):
            for token_id in group:
                input_ids.append(token_id)
                group_map.append(group_idx)
        
        # 3. Truncation and Padding (Dynamic)
        input_ids = input_ids[:self.max_len]
        group_map = group_map[:self.max_len]
        
        attention_mask = [1] * len(input_ids)
        if len(input_ids) < self.max_len:
            pad_len = self.max_len - len(input_ids)
            input_ids += [self.tokenizer.pad_token_id] * pad_len
            group_map += [-1] * pad_len # -1 for padding positions
            attention_mask += [0] * pad_len
            
        input_ids = torch.tensor(input_ids)
        attention_mask = torch.tensor(attention_mask)
        labels = input_ids.clone()
        
        # 4. Whole Word Masking Logic (Dynamic per call)
        # Identify unique word groups (excluding padding)
        unique_groups = list(set([g for g in group_map if g != -1]))
        
        # Decide which groups to mask (15% probability)
        masked_group_indices = [g for g in unique_groups if random.random() < self.mask_prob]
        
        # Apply masks to all tokens in the selected groups
        masked_any = False
        for i, g_idx in enumerate(group_map):
            if g_idx in masked_group_indices:
                input_ids[i] = self.mask_token_id
                masked_any = True
            else:
                labels[i] = -100 # Ignore in loss
        
        # Edge case: if no tokens were masked, randomly mask at least one non-special token
        if not masked_any and len(unique_groups) > 0:
            rand_group = random.choice(unique_groups)
            for i, g_idx in enumerate(group_map):
                if g_idx == rand_group:
                    input_ids[i] = self.mask_token_id
                    labels[i] = input_ids[i] # Original ID for prediction
                else:
                    labels[i] = -100
        
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
