import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from model_scratch import SimpleTransformer
from tokenizer_bpe import BPETokenizer
from dataset import WaimaiDataset
import os
import numpy as np

# Configuration
MODEL_PATH = 'best_model_finetuned.bin'
VOCAB_FILE = 'vocab_bpe.json'
DATA_FILE = 'online_shopping_10_cats.csv'
MAX_LEN = 128
BATCH_SIZE = 64
D_MODEL = 512
NUM_HEADS = 8
NUM_LAYERS = 4
D_FF = 2048
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_swa_state_dict(path):
    """Helper to load state_dict saved from SWA AveragedModel"""
    state_dict = torch.load(path, map_location=DEVICE)
    if 'n_averaged' in state_dict:
        del state_dict['n_averaged']
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    return new_state_dict

def evaluate():
    print(f"Evaluating model: {MODEL_PATH}")
    print(f"Using device: {DEVICE}")

    # Load Tokenizer
    tokenizer = BPETokenizer.load_vocab(VOCAB_FILE)
    vocab_size = tokenizer.vocab_size_actual
    print(f"Vocab size: {vocab_size}")

    # Initialize Model
    model = SimpleTransformer(
        vocab_size=vocab_size,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        d_ff=D_FF,
        num_classes=2
    )

    if not os.path.exists(MODEL_PATH):
        print(f"Error: {MODEL_PATH} not found.")
        return

    state_dict = load_swa_state_dict(MODEL_PATH)
    model.load_state_dict(state_dict)
    model = model.to(DEVICE)
    model.eval()

    # Prepare dataset (use same split seed as training if possible, but random 20% is standard)
    dataset = WaimaiDataset(DATA_FILE, tokenizer, MAX_LEN, augment=False)
    
    # We use a fixed seed for evaluation consistency
    torch.manual_seed(42)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    _, val_dataset = random_split(dataset, [train_size, val_size])
    
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    correct = 0
    total = 0
    
    print(f"Starting evaluation on {len(val_dataset)} samples...")
    
    with torch.no_grad():
        for d in val_loader:
            input_ids = d["input_ids"].to(DEVICE)
            attention_mask = d["attention_mask"].to(DEVICE)
            labels = d["labels"].to(DEVICE)
            
            logits = model(input_ids, attention_mask=attention_mask)
            preds = torch.argmax(logits, dim=1)
            
            correct += torch.sum(preds == labels).item()
            total += labels.size(0)

    accuracy = correct / total
    print("-" * 30)
    print(f"Final Accuracy: {accuracy:.4f}")
    print(f"Correct: {correct}/{total}")
    print("-" * 30)

if __name__ == "__main__":
    evaluate()
