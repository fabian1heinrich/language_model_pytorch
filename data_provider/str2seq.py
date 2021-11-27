import torch


def str2seq(seq_str, vocab, tokenizer, device=0):

    seq_token = [vocab(tokenizer(item)) for item in seq_str.split(' ')]
    seq = torch.tensor(seq_token, dtype=int, device=0)
    return seq