import torch
import torch.nn as nn

from data_provider import generate_square_subsequent_mask


def evaluate_model(model, eval_data):

    model.eval()
    eval_loss = 0.0

    criterion = nn.CrossEntropyLoss()
    src_mask = generate_square_subsequent_mask(eval_data.seq_len)

    with torch.no_grad():
        for inputs, targets, index in eval_data:
            outputs = model(inputs, src_mask)
            loss = criterion(outputs, targets)

            eval_loss += loss.item()

        eval_loss /= len(eval_data)

    return eval_loss