import torch


class Embedding(torch.nn.Module):
    def __init__(self, vocab_size, d_model, device=None, dtype=None):
        super().__init__()
        self.num_embeddings = vocab_size
        self.embedding_dim = d_model
        self.device = device
        self.dtype = dtype

        E =  torch.empty(vocab_size, d_model, dtype=self.dtype)
        E_tensor = torch.nn.init.trunc_normal_(E)
        E_tensor = E_tensor.to(self.device)
        self.embedding = torch.nn.Parameter(E_tensor)

    def forward(self, token_ids:torch.Tensor):
        #toke_ids in shape [batch, seq_length], which is 2D
        #returns a result in shape [batch, seq, d_model]
        return self.embedding[token_ids]
