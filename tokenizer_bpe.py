import json
import os
from collections import Counter, defaultdict
import re
from tqdm import tqdm

class BPETokenizer:
    def __init__(self, vocab_size=5000):
        self.vocab_size = vocab_size
        self.merges = []  # List of (pair, new_token) tuples
        self.vocab = {}   # token -> id mapping
        self.id_to_token = {}  # id -> token mapping
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.pad_token_id = 0
        self.unk_token_id = 1
        
    def _get_stats(self, word_freqs):
        """Count frequency of adjacent character pairs."""
        pairs = Counter()
        for word, freq in word_freqs.items():
            for i in range(len(word) - 1):
                pairs[(word[i], word[i + 1])] += freq
        return pairs
    
    def _merge_pair(self, pair, word_freqs):
        """Merge the most frequent pair in the vocabulary."""
        new_word_freqs = {}
        bigram = pair
        replacement = ''.join(pair)
        
        for word, freq in word_freqs.items():
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == bigram[0] and word[i+1] == bigram[1]:
                    new_word.append(replacement)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word_freqs[tuple(new_word)] = freq
        return new_word_freqs
    
    def fit_on_texts(self, texts):
        """Learn BPE merges from a list of texts."""
        # Step 1: Initialize with character-level tokens (represented as tuples)
        word_freqs = Counter()
        print("Pre-processing texts for BPE...")
        for text in tqdm(texts, desc="Grouping texts"):
            if isinstance(text, str) and text.strip():
                chars = tuple(list(text.strip()))
                word_freqs[chars] += 1
        
        # Step 2: Learn merges
        print(f"Learning BPE merges (target vocab size: {self.vocab_size})...")
        # Estimate how many merges we need
        initial_chars = set()
        for word in word_freqs.keys():
            for char in word:
                initial_chars.add(char)
        
        current_vocab_size = len(initial_chars) + 2 # +2 for PAD and UNK
        num_merges = self.vocab_size - current_vocab_size
        
        if num_merges <= 0:
            print("Initial characters already exceed target vocab size.")
            num_merges = 0
        
        pbar = tqdm(total=num_merges, desc="Learning Merges")
        for i in range(num_merges):
            pairs = self._get_stats(word_freqs)
            if not pairs:
                break
                
            best_pair = max(pairs, key=pairs.get)
            word_freqs = self._merge_pair(best_pair, word_freqs)
            self.merges.append(best_pair)
            pbar.update(1)
        pbar.close()
        
        # Step 3: Build vocabulary
        self.vocab = {self.pad_token: 0, self.unk_token: 1}
        idx = 2
        
        # Add all tokens that appear after merging
        tokens_found = set()
        for word in word_freqs.keys():
            for token in word:
                tokens_found.add(token)
        
        # Sort tokens to ensure deterministic vocab order (optional but good)
        for token in sorted(list(tokens_found)):
            if token not in self.vocab:
                self.vocab[token] = idx
                idx += 1
        
        self.id_to_token = {v: k for k, v in self.vocab.items()}
        print(f"Vocabulary built! Size: {len(self.vocab)} (learned {len(self.merges)} merges)")
    
    def _apply_merges(self, text):
        """Apply learned merges to encode text."""
        # Start with character-level (as a list)
        tokens = list(text)
        
        # Apply each merge in order
        for pair in self.merges:
            replacement = ''.join(pair)
            i = 0
            new_tokens = []
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i+1] == pair[1]:
                    new_tokens.append(replacement)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
        return tokens
    
    def encode(self, text, max_len=128):
        """Encode text to list of token IDs."""
        if not isinstance(text, str):
            text = str(text)
        
        # Apply BPE
        tokens = self._apply_merges(text)
        
        # Convert to IDs
        ids = [self.vocab.get(token, self.unk_token_id) for token in tokens]
        
        # Truncate
        ids = ids[:max_len]
        
        # Pad
        if len(ids) < max_len:
            ids += [self.pad_token_id] * (max_len - len(ids))
        
        return ids
    
    def decode(self, ids):
        """Decode token IDs back to text."""
        tokens = []
        for i in ids:
            if i == self.pad_token_id:
                continue
            tokens.append(self.id_to_token.get(i, self.unk_token))
        return ''.join(tokens)
    
    def save_vocab(self, filepath):
        """Save vocabulary and merges."""
        data = {
            'vocab': self.vocab,
            'merges': self.merges,
            'vocab_size': self.vocab_size
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"BPE vocabulary saved to {filepath}")
    
    @classmethod
    def load_vocab(cls, filepath):
        """Load vocabulary and merges."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        tokenizer = cls(vocab_size=data['vocab_size'])
        tokenizer.vocab = data['vocab']
        tokenizer.merges = [tuple(pair) for pair in data['merges']]
        tokenizer.id_to_token = {v: k for k, v in tokenizer.vocab.items()}
        return tokenizer
    
    @property
    def vocab_size_actual(self):
        return len(self.vocab)
    
    def __call__(self, text, max_length=128, padding='max_length', truncation=True, return_tensors=None, **kwargs):
        """Compatibility layer to mimic transformers tokenizer."""
        encoded_ids = self.encode(text, max_len=max_length)
        
        output = {
            'input_ids': encoded_ids,
            'attention_mask': [1 if i != self.pad_token_id else 0 for i in encoded_ids]
        }
        
        if return_tensors == 'pt':
            import torch
            # Add batch dimension [1, seq_len] to mimic standard tokenizers
            output['input_ids'] = torch.tensor(output['input_ids']).unsqueeze(0)
            output['attention_mask'] = torch.tensor(output['attention_mask']).unsqueeze(0)
        
        return output
