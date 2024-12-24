import json
from torch.utils.data import Dataset
import random

def load_data(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data

class TextDataset(Dataset):
    def __init__(self, standard_sentences, kansai_sentences, tokenizer, max_length=100):
        self.standard_sentences = standard_sentences
        self.kansai_sentences = kansai_sentences
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return min(len(self.standard_sentences), len(self.kansai_sentences))
    
    def __getitem__(self, idx):
        standard_encoding = self.tokenizer(self.standard_sentences[idx], truncation=True, padding='max_length', max_length=self.max_length, return_tensors="pt")
        kansai_encoding = self.tokenizer(self.kansai_sentences[idx], truncation=True, padding='max_length', max_length=self.max_length, return_tensors="pt")
        
        return {
            'A': standard_encoding['input_ids'].squeeze(0),
            'B': kansai_encoding['input_ids'].squeeze(0),
        }

class TextDataset_V2(Dataset):
    def __init__(self, standard_sentences, kansai_sentences, tokenizer, max_length=128):
        self.standard_sentences = standard_sentences
        self.kansai_sentences = kansai_sentences
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        while len(self.standard_sentences) < len(self.kansai_sentences):
            self.standard_sentences.append(random.choice(self.standard_sentences))
        while len(self.kansai_sentences) < len(self.standard_sentences):
            self.kansai_sentences.append(random.choice(self.kansai_sentences))
        
    def __len__(self):
        return max(len(self.standard_sentences), len(self.kansai_sentences))
    
    def __getitem__(self, idx):
        standard_encoding = self.tokenizer(self.standard_sentences[idx], truncation=True, padding='max_length', max_length=self.max_length, return_tensors="pt")
        kansai_encoding = self.tokenizer(self.kansai_sentences[idx], truncation=True, padding='max_length', max_length=self.max_length, return_tensors="pt")
        
        return {
            'A': standard_encoding['input_ids'].squeeze(0),
            'B': kansai_encoding['input_ids'].squeeze(0),
        }
    

def get_text_embedding(src, embedding_layer):
    outputs = embedding_layer(src)
    bert_emb = outputs.last_hidden_state  # [batch_size, seq_len, 768]
    bert_emb = bert_emb.reshape(bert_emb.size(0), bert_emb.size(1), 256, 3)  # [batch_size, seq_len, 256, 3]
    emb_256 = bert_emb.mean(dim=-1)  # [batch_size, seq_len, 256]
    return emb_256