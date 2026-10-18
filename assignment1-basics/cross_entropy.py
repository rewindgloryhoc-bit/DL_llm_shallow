
import torch


def cross_entropy(logits, target_tokens):
    #note logits are in shape [(batch seq) vocab_size]
    #the second argument is then in shape [(batch seq)]
    # Softmax / CE are precision-sensitive: always compute in fp32.
    logits = logits.float()
    c = torch.max(logits, dim=-1, keepdim=True).values #sum at vocab axis
    l_norm = torch.exp(logits - c)
    #calc [n] matrix. save for space
    first_term = torch.log(torch.sum(l_norm, dim=-1)) + c.squeeze(-1)
    #prepare logits in [batch, seq]
    target_logits = torch.gather(logits, dim=-1, index=target_tokens.unsqueeze(-1)).squeeze(-1)
    loss =  first_term - target_logits
    return torch.mean(loss)


