import torch
import torch.nn as nn
from flash_attention import MHA
from util import get_text_embedding

class Generator(nn.Module):
    def __init__(self, vocab_size, embedding_layer, d_model=256, nhead=8, num_encoder_layers=1, num_decoder_layers=1):
        super(Generator, self).__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size

        self.embedding = embedding_layer
        self.encoder = MHA(d_model, nhead, dropout=0.1)
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead),
            num_layers=num_decoder_layers
        )
        self.output_layer = nn.Linear(d_model, vocab_size)
        
    def forward(self, src):
        """
        src: [batch_size, seq_len]
        """
        src = get_text_embedding(src, self.embedding)  # [batch_size, seq_len, d_model]
        src = src.permute(1, 0, 2)  # [src_len, batch_size, d_model]
        memory = self.encoder(src)  # [src_len, batch_size, d_model]
        tgt = torch.zeros(src.size(1), 1).long().to(src.device)  # [batch_size, 1]

        generated_tokens = []
        for _ in range(100):
            tgt_emb = get_text_embedding(tgt, self.embedding)  # [batch_size, tgt_len, d_model]
            tgt_emb = tgt_emb.permute(1, 0, 2)  # [tgt_len, batch_size, d_model]
            output = self.decoder(tgt_emb, memory)  # [tgt_len, batch_size, d_model]
            logits = self.output_layer(output)  # [tgt_len, batch_size, vocab_size]
            probs = torch.softmax(logits, dim=-1) # [tgt_len, batch_size, vocab_size]
            next_token = torch.argmax(probs, dim=-1)[-1]  # [batch_size]
            generated_tokens.append(next_token.unsqueeze(1)) # [batch_size, 1]
            tgt = torch.cat([tgt, next_token.unsqueeze(1)], dim=1) # [batch_size, tgt_len]
        
        return torch.cat(generated_tokens, dim=1)
    

class Discriminator(nn.Module):
    def __init__(self, vocab_size, embedding_layer, d_model=256, nhead=8, num_layers=1):
        super(Discriminator, self).__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size

        self.embedding = embedding_layer
        self.transformer_encoder = MHA(d_model, nhead, dropout=0.1)
        self.fc1 = nn.Linear(d_model, 256)
        self.fc2 = nn.Linear(256, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        """
        x: [batch_size, seq_len]
        """
        x = get_text_embedding(x, self.embedding)
        x = x.permute(1, 0, 2)  # [seq_len, batch_size, d_model]
        x = self.transformer_encoder(x)  # [seq_len, batch_size, d_model]
        x = x.mean(dim=0)  # [batch_size, d_model]
        x = torch.relu(self.fc1(x))  # [batch_size, 256]
        x = self.fc2(x)  # [batch_size, 1]
        
        prob = self.sigmoid(x)  # [batch_size, 1]
        
        return prob