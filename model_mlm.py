import torch
import torch.nn as nn
import math

# Import the existing SimpleTransformer components
from model_scratch import PositionalEncoding, EncoderLayer

class TransformerMLM(nn.Module):
    """Transformer for Masked Language Modeling (Pre-training)"""
    def __init__(self, vocab_size, d_model=512, num_heads=8, num_layers=4, d_ff=2048, max_len=128):
        super(TransformerMLM, self).__init__()
        self.d_model = d_model
        
        # Shared components with SimpleTransformer
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len + 1)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff) for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model) # Final norm for Pre-Norm architecture
        self.dropout = nn.Dropout(0.1)
        
        # MLM head: predict which token was masked
        self.mlm_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, vocab_size)
        )
        
    def forward(self, input_ids, attention_mask=None):
        batch_size = input_ids.size(0)
        
        # 1. Embed and scaling
        x = self.embedding(input_ids) * math.sqrt(self.d_model)
        
        # 2. Prepend [CLS] token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1) # [batch, seq+1, d_model]
        
        # 3. Adjust mask for [CLS] token
        if attention_mask is not None:
            cls_mask = torch.ones(batch_size, 1).to(attention_mask.device)
            full_mask = torch.cat((cls_mask, attention_mask), dim=1)
            mask = full_mask.unsqueeze(1).unsqueeze(2)
        else:
            mask = None
            
        x = self.pos_encoder(x)
        x = self.dropout(x)
        
        for layer in self.layers:
            x = layer(x, mask)
        
        x = self.norm(x) # Final norm
        
        # Predict tokens at all positions (including CLS, but we can ignore it later)
        logits = self.mlm_head(x)
        return logits
    
    def save_encoder(self, path):
        """Save encoder weights for transfer learning"""
        state_dict = {
            'embedding': self.embedding.state_dict(),
            'pos_encoder': self.pos_encoder.state_dict(),
            'cls_token': self.cls_token,
            'layers': [layer.state_dict() for layer in self.layers],
            'norm': self.norm.state_dict(),
            'dropout': self.dropout.state_dict()
        }
        torch.save(state_dict, path)
        print(f"Encoder weights saved to {path}")
    
    @staticmethod
    def load_encoder_to_classifier(encoder_path, classifier_model):
        """Load pre-trained encoder into a SimpleTransformer classifier"""
        state_dict = torch.load(encoder_path)
        
        classifier_model.embedding.load_state_dict(state_dict['embedding'])
        classifier_model.pos_encoder.load_state_dict(state_dict['pos_encoder'])
        classifier_model.cls_token.data = state_dict['cls_token'].data
        
        for i, layer_state in enumerate(state_dict['layers']):
            classifier_model.layers[i].load_state_dict(layer_state)
        
        classifier_model.norm.load_state_dict(state_dict['norm'])
        classifier_model.dropout.load_state_dict(state_dict['dropout'])
        print(f"Encoder weights loaded from {encoder_path}")
        return classifier_model
