import torch


class MLP(torch.nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        
        self.FC1 = torch.nn.Linear(input_dim, hidden_dim)
        self.gelu = torch.nn.GELU()
        self.FC2 = torch.nn.Linear(hidden_dim, output_dim) 

    def forward(self, x):

        if x.dim() == 3:
            # forward on multiple mutation to compute additive score, x: (mutation_num, batch_size, input_dim)
            mutation_num, batch_size, input_dim = x.size()
            x = x.reshape(-1, input_dim)  # (mutation_num * batch_size, seq_len)
            reshape_needed = True
        else:
            # forward on single mutation or multiple mutation, x: (batch_size, seq_len)
            reshape_needed = False

        logits = self.FC2(self.gelu(self.FC1(x)))
    
        if reshape_needed:
            return logits.view(mutation_num, batch_size, -1)  # (mutation_num, batch_size, output_dim)
        else:
            return logits 
    