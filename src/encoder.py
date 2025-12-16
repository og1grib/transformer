import torch
import torch.nn as nn

from src.attention import MultiheadAttention


class EncoderBlock(nn.Module):
    
    def __init__(
        self,
        input_size,                            # размерность входных эмбеддингов
        output_size,                           # размерность выходных эмбеддингов
        
        num_heads,                             # количество голов внимания
        head_size,                             # размерность голов внимания
        
        hidden_size,                           # размерность скрытого слоя feed forward
        query_cross_attention_size=None        # размерность входных эмбеддингов для query (кросс-внимания)
        ):
        
        super().__init__()
        
        query_input_size = input_size if query_cross_attention_size is None else query_cross_attention_size
        
        self.attention = MultiheadAttention(
            input_size=input_size,
            output_size=output_size,
            num_heads=num_heads,
            head_size=head_size,
            query_cross_attention_size=query_input_size
            )
        
        self.norm_1 = nn.LayerNorm(output_size)
        self.norm_2 = nn.LayerNorm(output_size)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(output_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

        if query_input_size != output_size:
            self.adapt_query = nn.Linear(query_input_size, output_size)
        else:
            self.adapt_query = nn.Identity()
        
    def forward(self, query, key, value):
        # q, k, v [batch_size, seq_len, emb_size]
        
        # Multihead Attention
        att_out = self.attention(query, key, value)
        
        # Add & Norm
        out = att_out + self.adapt_query(query)
        out = self.norm_1(out)
        
        # Feed Forward
        ff_out = self.feed_forward(out)
        
        # Add & Norm
        out = out + ff_out
        out = self.norm_2(out)
        
        return out
    

class EncoderTransformer(nn.Module):
    
    def __init__(
        self,
        N,                         # количество блоков энкодера
        input_size,                # размерность входных эмбеддингов
        output_size,               # размерность выходных эмбеддингов
        
        num_heads,                 # количество голов внимания
        head_size,                 # размерность голов внимания
        
        hidden_size                # размерность скрытого слоя feed forward
        ):
        
        super().__init__()
        
        self.encoder_blocks = nn.ModuleDict({
            f"encoder_block_{i}": EncoderBlock(
                input_size=input_size if i==0 else output_size,
                output_size=output_size, 
                head_size=head_size,
                num_heads=num_heads,
                hidden_size=hidden_size)
            for i in range(N)
        })
        
        
    def forward(self, x):
        # [batch_size, seq_len, input_size]
        out = x
        
        for i, layer in self.encoder_blocks.items():
            out = layer(out, out, out)
            
        # [batch_size, seq_len, output_size]
        return out