import torch
import torch.nn as nn
import os
from model_scratch import SimpleTransformer
from tokenizer_bpe import BPETokenizer

# Configuration (Must match training)
MODEL_PATH = 'best_model_finetuned.bin'
VOCAB_FILE = 'vocab_bpe.json'
MAX_LEN = 128 # Input sequence length
D_MODEL = 512
NUM_HEADS = 8
NUM_LAYERS = 4
D_FF = 2048
DEVICE = torch.device("cpu") # Export on CPU for universal compatibility

def load_swa_state_dict(path):
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

def export():
    print("Loading model for export...")
    tokenizer = BPETokenizer.load_vocab(VOCAB_FILE)
    
    model = SimpleTransformer(
        vocab_size=tokenizer.vocab_size_actual,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        d_ff=D_FF,
        num_classes=2
    )
    
    if os.path.exists(MODEL_PATH):
        state_dict = load_swa_state_dict(MODEL_PATH)
        model.load_state_dict(state_dict)
        print("Weights loaded.")
    else:
        print(f"Error: {MODEL_PATH} not found.")
        return

    model.eval()
    
    # Create dummy inputs for tracing
    dummy_input = torch.randint(0, tokenizer.vocab_size_actual, (1, MAX_LEN))
    dummy_mask = torch.ones(1, MAX_LEN)
    
    print("\n1. Exporting to TorchScript...")
    try:
        # TorchScript tracing
        traced_model = torch.jit.trace(model, (dummy_input, dummy_mask))
        traced_model.save("model_torchscript.pt")
        print("Success: model_torchscript.pt saved.")
    except Exception as e:
        print(f"TorchScript export failed: {e}")

    print("\n1. Exporting to TorchScript...")
    try:
        # TorchScript tracing
        traced_model = torch.jit.trace(model, (dummy_input, dummy_mask))
        traced_model.save("model_torchscript.pt")
        print("Success: model_torchscript.pt saved.")
    except Exception as e:
        print(f"TorchScript export failed: {e}")

if __name__ == "__main__":
    export()
