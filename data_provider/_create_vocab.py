from torchtext.vocab import build_vocab_from_iterator


def create_vocab(raw_text_iter, tokenizer):
    vocab = build_vocab_from_iterator(
        map(tokenizer, raw_text_iter),
        specials=['<unk>'],
    )
    vocab.set_default_index(vocab['<unk>'])

    return vocab