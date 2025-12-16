import torch
from src.transformer import Transformer


def test_transformer_forward_shape():
    
    model = Transformer(
        n=50,
        emb_size=18,
        max_seq=20,

        N_enc=3,
        output_size_enc=16,
        num_heads_enc=4,
        head_size_enc=10,
        hidden_size_enc=35,

        N_dec=5,
        output_size_dec=30,
        num_heads_dec=2,
        head_size_dec=14,
        hidden_size_dec=36,

        output_size=100
    )

    batch = 4
    seq_len = 10

    x_enc = torch.randint(0, 50, (batch, seq_len))
    x_dec = torch.randint(0, 50, (batch, seq_len))

    out, _ = model(x_enc, x_dec)

    assert out.shape == (batch, seq_len, 100)
