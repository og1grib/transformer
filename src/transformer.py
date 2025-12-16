import torch
import torch.nn as nn

from src.positional_enc import PositionalEncoding
from src.encoder import EncoderTransformer
from src.decoder import DecoderTransformer


class Transformer(nn.Module):
    
    def __init__(
        self,
        n,                 # размер словаря
        emb_size,          # размер эмбеддингов
        max_seq,           # максимальная длина последовательности
        
        N_enc, output_size_enc, num_heads_enc, head_size_enc, hidden_size_enc,
        N_dec, output_size_dec, num_heads_dec, head_size_dec, hidden_size_dec,
        
        output_size,        # размер выходных эмбеддингов
        ):
        
        super().__init__()
        
        self.enc_embedding = nn.Embedding(n, emb_size)
        self.dec_embedding = nn.Embedding(n, emb_size)
        
        self.enc_pos = PositionalEncoding(max_seq, emb_size)
        self.dec_pos = PositionalEncoding(max_seq, emb_size)
        
        self.encoder = EncoderTransformer(
            N_enc, 
            emb_size,
            output_size_enc, 
            num_heads_enc, 
            head_size_enc, 
            hidden_size_enc
            )
        
        self.decoder = DecoderTransformer(
            N_dec,
            emb_size,
            output_size_dec,
            num_heads_dec,
            head_size_dec,
            hidden_size_dec,
            encoder_output_size=output_size_enc
            )
    
        self.linear = nn.Linear(output_size_dec, output_size)
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x_enc, x_dec):
        # [batch, seq_len]
        enc_emb = self.enc_embedding(x_enc)
        dec_emb = self.dec_embedding(x_dec)
        
        # [batch, seq_len, emb_size]
        enc_pos = self.enc_pos(enc_emb) 
        dec_pos = self.dec_pos(dec_emb)
        
        # [batch, seq_len, output_size_enc]
        enc_out = self.encoder(enc_pos)
        dec_out = self.decoder(dec_pos, enc_out)
        
        out = self.linear(dec_out)
        # out = self.softmax(out)
        
        return out
        