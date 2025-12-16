import torch
import torch.nn as nn
from src.attention import MultiheadAttention
from src.encoder import EncoderBlock

class DecoderBlock(nn.Module):
    
    def __init__(
        self,
        input_size,                            # размерность входных эмбеддингов
        output_size,                           # размерность выходных эмбеддингов
        
        num_heads,                             # количество голов внимания
        head_size,                             # размерность голов внимания
        
        hidden_size,                           # размерность скрытого слоя feed forward
        encoder_output_size,                   # размерность выходных эмбеддингов энкодера
        ):
        
        super().__init__()
        
        self.masked_attention = MultiheadAttention(
            input_size=input_size,
            output_size=output_size,
            num_heads=num_heads,
            head_size=head_size,
            masked=True)
        
        if input_size != output_size:
            self.adapt = nn.Linear(input_size, output_size)
        else:
            self.adapt = nn.Identity()
            
        self.norm_1 = nn.LayerNorm(output_size)
        
        # self.enc_like_block = EncoderBlock(
        #     encoder_output_size, # key, value
        #     output_size,
        #     num_heads,
        #     head_size,
        #     hidden_size,
        #     query_cross_attention_size=output_size # query
        #     ) 
        
        self.cross_attention = MultiheadAttention(
            input_size=encoder_output_size,     # key/value
            output_size=output_size,
            num_heads=num_heads,
            head_size=head_size,
            query_cross_attention_size=output_size,
            return_attn=True
        )

        self.norm_2 = nn.LayerNorm(output_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(output_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

        self.norm_3 = nn.LayerNorm(output_size)

        
    def forward(self, x, enc_out):
        # [batch_size, seq_len, input_size]
        
        # Masked multi-head attention
        mask_att_out = self.masked_attention(x, x, x)
        
        # Add & Norm
        x = mask_att_out + self.adapt(x)
        x = self.norm_1(x)
        
        # Encoder-Decoder Attention
        cross_out, cross_attn = self.cross_attention(
            x, enc_out, enc_out
        )

        # Add & Norm
        x = x + cross_out
        x = self.norm_2(x)
        
        # Feed-forward
        ff_out = self.feed_forward(x)
        
        # Add & Norm
        x = x + ff_out
        x = self.norm_3(x)

        return x, cross_attn

        
class DecoderTransformer(nn.Module):
    
    def __init__(
        self,
        N,                         # количество блоков декодера
        input_size,                # размерность входных эмбеддингов
        output_size,               # размерность выходных эмбеддингов
        
        num_heads,                 # количество голов внимания
        head_size,                 # размерность голов внимания
        
        hidden_size,               # размерность скрытого слоя feed forward
        encoder_output_size=None      # размерность выходных эмбеддингов энкодера
        ):
        
        super().__init__()
        
        self.decoder_blocks = nn.ModuleDict({
            f"decoder_block_{i}": DecoderBlock(
                input_size=input_size if i==0 else output_size,
                head_size=head_size,
                num_heads=num_heads,
                output_size=output_size,
                hidden_size=hidden_size,
                encoder_output_size=encoder_output_size,
            ) for i in range(N)
        })
        
    def forward(self, x, enc_out):
        # x: [batch_size, seq_len, input_size]
        out = x
        attn = None
            
        for block in self.decoder_blocks.values():
            out, attn = block(out, enc_out)
        
        # x: [batch_size, seq_len, output_size]
        return out, attn
    