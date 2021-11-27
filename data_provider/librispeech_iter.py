import torchaudio

from torch.utils.data import IterableDataset


class LibriSpeechRawIter(IterableDataset):
    def __init__(self, data_mode: str = 'train'):
        super().__init__()

        self.counter = 0
        if data_mode == 'train':
            self.libri_data = torchaudio.datasets.LIBRISPEECH(
                '.libri/',
                url='train-clean-100',
                download=True,
            )
        elif data_mode == 'test':
            self.libri_data = torchaudio.datasets.LIBRISPEECH(
                '.libri/',
                url='test-clean',
                download=True,
            )
        else:
            # actually it should raise an exception
            print('no data mode given')

    def __iter__(self):
        self.counter = 0
        return self

    def __next__(self):

        _, _, utterance, _, _, _ = self.libri_data.__getitem__(self.counter)

        self.counter += 1
        if self.counter >= len(self.libri_data):
            raise StopIteration()

        return utterance

    def __len__(self):
        return len(self.libri_data)