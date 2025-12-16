import torch

from src.attention import MultiheadAttention
from src.encoder import EncoderBlock
from src.encoder import EncoderTransformer
from src.positional_enc import PositionalEncoding

def test_attention_output_shape():
    attn = MultiheadAttention(
        input_size=16,
        output_size=32,
        num_heads=4,
        head_size=8
    )

    x = torch.randn(2, 10, 16)
    out = attn(x, x, x)

    assert out.shape == (2, 10, 32)


def test_cross_attention_shape():
    attn = MultiheadAttention(
        input_size=16,
        output_size=16,
        num_heads=2,
        head_size=8,
        query_cross_attention_size=32
    )

    query = torch.randn(2, 5, 32)
    key   = torch.randn(2, 10, 16)
    value = torch.randn(2, 10, 16)

    out = attn(query, key, value)

    assert out.shape == (2, 5, 16)


def test_encoder_block_output_shape():
    block = EncoderBlock(
        input_size=16,
        output_size=32,
        num_heads=4,
        head_size=8,
        hidden_size=64
    )

    x = torch.randn(2, 10, 16)
    out = block(x, x, x)

    assert out.shape == (2, 10, 32)

def test_encoder_block_query_adaptation():
    block = EncoderBlock(
        input_size=16,
        output_size=32,
        num_heads=2,
        head_size=8,
        hidden_size=64,
        query_cross_attention_size=16
    )

    x = torch.randn(1, 5, 16)
    out = block(x, x, x)

    assert out.shape[-1] == 32


def test_encoder_transformer_output_shape():
    model = EncoderTransformer(
        N=3,
        input_size=16,
        output_size=32,
        num_heads=4,
        head_size=8,
        hidden_size=64
    )

    x = torch.randn(2, 10, 16)
    out = model(x)

    assert out.shape == (2, 10, 32)


def test_positional_encoding_shape():
    pe = PositionalEncoding(max_len=50, d_model=16)

    x = torch.zeros(2, 10, 16)
    out = pe(x)

    assert out.shape == x.shape

