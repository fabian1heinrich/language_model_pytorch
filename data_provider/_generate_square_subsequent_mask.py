import torch


def generate_square_subsequent_mask(sz: int, device=0):

    square_subsequent_mask = torch.triu(
        torch.ones(sz, sz) * float('-inf'),
        diagonal=1,
    )

    return square_subsequent_mask.to(device)
