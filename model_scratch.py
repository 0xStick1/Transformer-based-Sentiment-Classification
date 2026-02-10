import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding. 
    Using sin/cos fixed patterns ensures the model can potentially generalize 
    to sequence lengths longer than those seen during training.
    """
    def __init__(self, d_model, max_len=512):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # x shape: [batch, seq_len, d_model]
        return x + self.pe[:, :x.size(1), :]

class MultiHeadAttention(nn.Module):
    """
    Standard Multi-Head Attention.
    Splitting d_model into num_heads allows the model to jointly attend to 
    information from different representation subspaces at different positions.
    """
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Linear projections & split into heads
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled Dot-Product Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        attn = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn, V).transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.W_o(context), attn

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(0.1)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU() # Standard ReLU; GELU often preferred for pre-training but ReLU is stable here.

    def forward(self, x):
        return self.linear2(self.dropout(self.relu(self.linear1(x))))

class EncoderLayer(nn.Module):
    """
    Transformer Encoder Block with Pre-LayerNorm.
    Research indicates Pre-Norm improves gradient flow and stability in deeper models, 
    allowing for more aggressive learning rates.
    """
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Pre-LayerNorm flow: Norm -> Attention -> Residual
        x2 = self.norm1(x)
        attn_output, _ = self.self_attn(x2, x2, x2, mask)
        x = x + self.dropout(attn_output)
        
        # Pre-LayerNorm flow: Norm -> FFN -> Residual
        x2 = self.norm2(x)
        ffn_output = self.ffn(x2)
        x = x + self.dropout(ffn_output)
        return x

class SimpleTransformer(nn.Module):
    """
    The main Transformer Classifier.
    Updates in v3.0: 
    1. Switched to [CLS] token based pooling for global semantic capture.
    2. Adopted Pre-LayerNorm for training robustness.
    """
    def __init__(self, vocab_size, d_model=256, num_heads=4, num_layers=2, d_ff=512, num_classes=2, max_len=128):
        super(SimpleTransformer, self).__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len + 1) # Support for [CLS] + sequence
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model)) 
        
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff) for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model) # Final LN before classification
        self.dropout = nn.Dropout(0.1)
        
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, num_classes)
        )
        
    def forward(self, input_ids, attention_mask=None):
        batch_size = input_ids.size(0)
        
        # Scaling embeddings by sqrt(d_model) to keep variance stable
        x = self.embedding(input_ids) * math.sqrt(self.d_model)
        
        # Prepend learnable [CLS] token to input embeddings
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1) 
        
        # Adjust attention mask for the extra [CLS] position
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
            
        x = self.norm(x) 
        
        # Extract the final hidden state of the [CLS] token for sentiment logit prediction
        cls_out = x[:, 0, :] 
        
        logits = self.classifier(cls_out)
        return logits
