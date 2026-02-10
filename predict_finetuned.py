import torch
import torch.nn.functional as F
from model_scratch import SimpleTransformer
from tokenizer_bpe import BPETokenizer
import os
import math

# Configuration
MODEL_PATH = 'best_model_finetuned.bin'
VOCAB_FILE = 'vocab_bpe.json'
MAX_LEN = 128
D_MODEL = 512
NUM_HEADS = 8
NUM_LAYERS = 4
D_FF = 2048
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_swa_state_dict(path):
    """Helper to load state_dict saved from SWA AveragedModel"""
    state_dict = torch.load(path, map_location=DEVICE)
    # 1. Remove 'n_averaged' if exists
    if 'n_averaged' in state_dict:
        del state_dict['n_averaged']
    
    # 2. Strip 'module.' prefix
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    return new_state_dict

def predict(text, model, tokenizer):
    model.eval()
    encoding = tokenizer(
        text,
        max_length=MAX_LEN,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(DEVICE)
    attention_mask = encoding['attention_mask'].to(DEVICE)
    
    with torch.no_grad():
        logits = model(input_ids, attention_mask=attention_mask)
        probs = torch.softmax(logits, dim=1)
        prediction = torch.argmax(probs, dim=1).item()
        confidence = probs[0][prediction].item()
        
    return prediction, confidence

def main():
    print("Loading fine-tuned SimpleTransformer model...")
    
    # Load Tokenizer
    if not os.path.exists(VOCAB_FILE):
        print(f"Error: {VOCAB_FILE} not found. Run pretrain.py then finetune.py first.")
        return
        
    tokenizer = BPETokenizer.load_vocab(VOCAB_FILE)
    print(f"Loaded BPE vocab size: {tokenizer.vocab_size_actual}")
    
    # Initialize Model
    model = SimpleTransformer(
        vocab_size=tokenizer.vocab_size_actual,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        d_ff=D_FF,  # Must match training configuration
        num_classes=2
    )
    
    if os.path.exists(MODEL_PATH):
        try:
            state_dict = load_swa_state_dict(MODEL_PATH)
            model.load_state_dict(state_dict)
            print("Loaded fine-tuned model weights (SWA-Compatible).")
        except Exception as e:
            print(f"Error loading model: {e}")
            return
    else:
        print(f"Warning: {MODEL_PATH} not found. Using uninitialized weights.")
    
    model = model.to(DEVICE)
    model.eval()
    
    print("Model loaded. Enter text to classify (or 'q' to quit).")
    
    while True:
        text = input("Text: ")
        if text.lower() == 'q':
            break
            
        sentiment, confidence = predict(text, model, tokenizer)
        
        label = "Positive" if sentiment == 1 else "Negative"
        
        print(f"Prediction: {label} (Confidence: {confidence:.4f})")

if __name__ == "__main__":
    main()
