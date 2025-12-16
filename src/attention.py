import torch 
import torch.nn as nn

class MultiheadAttention(nn.Module):
    
    def __init__(
        self, 
        input_size,                        # размерность входных эмбеддингов
        output_size,                       # размерность выходных эмбеддингов

        num_heads,                         # количество голов внимания
        head_size,                         # размерность голов внимания    
            
        query_cross_attention_size=None,   # размерность входных эмбеддингов для query (кросс-внимания)
        masked=False                       # маскирование (декодер)
        ):
        
        super().__init__()
        
        self.num_heads = num_heads
        self.head_size = head_size
        
        query_input_size = input_size if query_cross_attention_size is None else query_cross_attention_size
        
        self.W_q = nn.Linear(query_input_size, num_heads * head_size, bias=False)
        self.W_k = nn.Linear(input_size, num_heads * head_size, bias=False)
        self.W_v = nn.Linear(input_size, num_heads * head_size, bias=False)
        
        self.feed_forward = nn.Linear(num_heads * head_size, output_size)
        
        self.masked = masked
        self.query_input_size = query_input_size
        
        
    def forward(self, query, key, value):
        # q, k, v [batch_size, seq_len, input_size]
        
        batch_size = key.size(0)
        seq_len = key.size(1)
        query_seq_len = query.size(1)

        q = self.W_q(query)
        k = self.W_k(key)
        v = self.W_v(value)
        # q, k, v [batch_size, seq_len, num_heads * head_size]
        
        q = q.view(batch_size, query_seq_len, self.num_heads, self.head_size).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_size).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_size).transpose(1, 2)
        # q, k, v [batch_size, num_heads, seq_len, head_size]
        
        k_T = k.transpose(2, 3)
        # k [batch_size, num_heads, head_size, seq_len]
        
        relevance = torch.matmul(q, k_T) / (self.head_size ** 0.5)
        # [batch_size, num_heads, query_seq_len, seq_len]
        
        if self.masked:
            mask = torch.tril(torch.ones((query_seq_len, seq_len))).to(torch.bool)
            relevance = relevance.masked_fill(~mask, float('-inf'))
        
        relevance = torch.softmax(relevance, dim=-1)
        heads = torch.matmul(relevance, v)
        # [batch_size, num_heads, query_seq_len, head_size]

        heads = heads.transpose(1, 2)
        # [batch_size, query_seq_len, num_heads, head_size]
        
        concat = heads.reshape(batch_size, query_seq_len, self.head_size * self.num_heads)  
 
        out = self.feed_forward(concat)
        # [batch_size, query_seq_len, output_size]
        
        return out
    