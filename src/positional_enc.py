import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    # Позиционное кодирование как в статье "Attention is All You Need" (https://arxiv.org/abs/1706.03762)
    
    def __init__(
        self,
        max_len,                             # максимальная длина последовательности
        d_model                              # размерность эмбеддингов
        ):
        
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len)[:, None]  # позиция токена
        i = torch.arange(0, d_model)[None, :]         # номер элемента позиционого вектора
        
        sin = torch.sin(position / (10000 ** (2 * (i[:, ::2]) / d_model)))
        cos = torch.cos(position / (10000 ** (2 * (i[:, 1::2]) / d_model)))
        pe[:, ::2] = sin
        pe[:, 1::2] = cos
        
        pe = pe[None, :, :]
        # [1, max_len, d_model]   
                                
        self.register_buffer('pe', pe)
        
        
    def forward(self, x):
        # [batch_size, emb_len, input_size]
        
        x_len = x.size(1)
        x = x + self.pe[:, :x_len, :]
        
        # [batch_size, emb_len, input_size]
        return x