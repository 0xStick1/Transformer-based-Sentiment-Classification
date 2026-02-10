import random
import torch
from torch.utils.data import Dataset
import pandas as pd

class SynonymAugmentor:
    """Simple synonym replacement for Chinese text augmentation"""
    def __init__(self):
        # Small dictionary of common Chinese sentiment synonyms
        self.synonyms = {
            "好": ["棒", "赞", "牛", "可以", "行", "不错"],
            "快": ["迅速", "给力", "极速"],
            "慢": ["延迟", "拖拉", "迟缓"],
            "差": ["烂", "糟糕", "不行", "劣质"],
            "贵": ["不划算", "奢侈", "高价"],
            "便宜": ["实惠", "亲民", "划算"],
            "推荐": ["力荐", "安利", "好评"],
            "难吃": ["难以下咽", "难喝", "槽糕"],
            "服务": ["态度", "接待"]
        }

    def augment(self, text):
        if not text:
            return text
        
        # 10% probability to replace a word with its synonym
        for word, syns in self.synonyms.items():
            if word in text and random.random() < 0.1:
                replacement = random.choice(syns)
                text = text.replace(word, replacement)
        return text

class WaimaiDataset(Dataset):
    def __init__(self, filename, tokenizer, max_len=128, augment=False):
        self.df = pd.read_csv(filename)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.augment = augment
        self.augmentor = SynonymAugmentor() if augment else None
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        review = str(self.df.iloc[idx]['review'])
        label = int(self.df.iloc[idx]['label'])
        
        # Apply augmentation if enabled (usually only for training)
        if self.augment and self.augmentor:
            review = self.augmentor.augment(review)
        
        encoding = self.tokenizer(
            review,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        
        return {
            'review_text': review,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }
