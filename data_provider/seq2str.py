def seq2str(seq, vocab):
    seq_str = ' '.join([vocab.lookup_token(item) for item in seq])
    return seq_str
