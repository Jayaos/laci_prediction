import torch


class SimpleLinear(torch.nn.Module):

    def __init__(self, input_dim, output_dim):
        super(SimpleLinear, self).__init__()
        
        self.FC = torch.nn.Linear(input_dim, output_dim) 

    def forward(self, x):

        if x.dim() == 3:
            # forward on multiple mutations to compute additive score, x: (mutation_num, batch_size, input_dim)
            mutation_num, batch_size, input_dim = x.size()
            x = x.reshape(-1, input_dim)  # (mutation_num * batch_size, seq_len)
            reshape_needed = True
        else:
            # evaluation on single mutation, x: (batch_size, seq_len)
            reshape_needed = False

        logits = self.FC(x)
    
        if reshape_needed:
            return logits.view(mutation_num, batch_size, -1)  # (mutation_num, batch_size, output_dim)
        else:
            return logits 
    