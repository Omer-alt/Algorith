import torch 
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_sequence_length) :
        super().__init__()
        self.max_sequence_length = max_sequence_length
        self.d_model = d_model
        
    def pe_encodding(self):
        
        i = torch.arange(0, self.d_model, 2).float()
        denominator = torch.pow(1000, i/self.d_model)
        position = torch.arange(self.max_sequence_length).reshape(self.max_sequence_length, 1)
        even_PE = torch.sin(position/denominator)
        odd_PE = torch.cos(position/denominator)
        stacked = torch.cat([even_PE.unsqueeze(2), odd_PE.unsqueeze(2)], dim=2) 
        PE = stacked.view(stacked.size(0), -1)
        
        return PE
    
pe = PositionalEncoding(d_model=6, max_sequence_length=10)
PE = pe.pe_encodding()
print(PE)