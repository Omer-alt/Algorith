import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiheadAttention(nn.Module):
    """_summary_

    Args:
        nn (_type_): _description_
        input_dim(int): 
    """
    def __init__(self, input_dim, d_model, num_heads):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.qkv_layer = nn.Linear(input_dim , 3 * d_model)
        self.linear_layer = nn.Linear(d_model, d_model)

    """_summary_: Compute Attention matrice 
    """
    def scaled_dot_product(self ,q, k, v, mask=None):

      d_k = q.size()[-1]
      scaled = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(d_k)

      mat_mask = self.get_mask(scaled)

      if mask:
          scaled += mat_mask
      attention = F.softmax(scaled, dim=-1)
      values = torch.matmul(attention, v)

      return values, attention

    """_summary_: Apply mask if we are in the case of decoder 
    """
    def get_mask(self, weight):

      mask = torch.full(weight.size(), float("-inf"))
      mask = torch.triu(mask, diagonal=1)

      return mask
    
    def forward(self, x, mask=None):
        batch_size, sequence_length, input_dim = x.size()
        
        qkv = self.qkv_layer(x)
        
        qkv = qkv.reshape(batch_size, sequence_length, self.num_heads, 3 * self.head_dim)
        print(f"qkv.size(): {qkv.size()}")
        
        # ordored in [batch_size, num_heads, sequence_length, 3 * head_dim]
        qkv = qkv.permute(0, 2, 1, 3)
        print(f"qkv.size(): {qkv.size()}")

        q, k, v = qkv.chunk(3, dim=-1)
        print(f"q size: {q.size()}, k size: {k.size()}, v size: {v.size()}, ")

        values, attention = self.scaled_dot_product(q, k, v, mask)
        print(f"values.size(): {values.size()}, attention.size:{ attention.size()} \n")
        
        if mask: 
            print(f"Masked Attention matrice :\n {attention}")
        else:
            print(f"Attention matrice  {attention}")

        values = values.reshape(batch_size, sequence_length, self.num_heads * self.head_dim)
        print(f"values.size(): {values.size()}")

        out = self.linear_layer(values)
        print(f"out.size(): {out.size()}")

        return out
    
if __name__ == "__main__":
    input_dim = 1024
    d_model = 512
    num_heads = 8

    batch_size = 30
    sequence_length = 5
    x = torch.randn( (batch_size, sequence_length, input_dim) )

    model = MultiheadAttention(input_dim, d_model, num_heads)
    out = model.forward(x, True)
    
    