import torch


def create_data_tensor(vocab, tokenizer, dataset_iter, batch_size):

    # create flat tensor
    data_tensor = [
        torch.tensor(vocab(tokenizer(item)), dtype=torch.long)
        for item in dataset_iter
    ]
    # remove empty tensors
    data_tensor = torch.cat(
        tuple(filter(lambda t: t.numel() > 0, data_tensor)), )

    # format to [length, batch_size]
    length = data_tensor.size(0) // batch_size
    data_tensor = data_tensor[:length * batch_size]
    data_tensor = data_tensor.view(batch_size, length).t().contiguous()

    return data_tensor


# if __name__ == '__main__':

#     from data_provider import LibriSpeechRawIter, create_vocab

#     raw_text_iter = LibriSpeechRawIter2()
#     tokenizer =
#     vocab = create_vocab(raw_text_iter, tokenizer)
#     dataset_iter =
#     batch_size =
#     print('run this')