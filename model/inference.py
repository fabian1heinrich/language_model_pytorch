import torch.nn as nn

from data_provider import generate_square_subsequent_mask


def inference(model, input):
    # input has to be of shape [x, 1]
    model.eval()

    src_mask = generate_square_subsequent_mask(input.size()[0])
    output = model(input, src_mask)

    output_prob = nn.functional.softmax(output, dim=1)

    return output_prob