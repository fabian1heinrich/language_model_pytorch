from torch.utils.data import IterableDataset
from torchtext.data.utils import get_tokenizer
from torchtext.datasets.wikitext2 import WikiText2

from data_provider import create_vocab, create_data_tensor


class LMData(IterableDataset):
    def __init__(
        self,
        vocab_iter,
        data_iter,
        batch_size,
        seq_len,
        device: int = 0,
    ):
        super().__init__()

        self.counter = 0
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.device = device

        self.tokenizer = get_tokenizer('basic_english')
        self.vocab = create_vocab(vocab_iter, self.tokenizer)
        self.n_token = len(self.vocab)

        self.data_tensor = create_data_tensor(
            self.vocab,
            self.tokenizer,
            data_iter,
            self.batch_size,
        )

    # call only data for 1 batch vs format to batch fomat prior?

    # returns complete batch
    def get_batch(self):
        index = self.counter
        seq_len = self.seq_len
        inputs = self.data_tensor[index * seq_len:(index + 1) * seq_len]
        targets = self.data_tensor[index * seq_len + 1:(index + 1) * seq_len +
                                   1]
        return (
            inputs.to(self.device),
            targets.reshape(-1).to(self.device),
            index,
        )

    def __iter__(self):
        self.counter = 0
        return self

    def __next__(self):
        self.counter += 1
        if self.counter >= len(self):
            raise StopIteration()
        return self.get_batch()

    def __len__(self):
        # drop last batch in case it isnt full size
        return self.data_tensor.size(0) // 35 - 1
