import math

from torchtext.datasets import WikiText103

from model import TransformerModel, train_model, evaluate_model, save_model
from data_provider import LMData

vocab_train = WikiText103(split='train')
vocab_test = WikiText103(split='train')
train_iter, eval_iter, test_iter = WikiText103()
train_data = LMData(vocab_train, train_iter, 20, 35)
test_data = LMData(vocab_test, test_iter, 20, 35)

n_token = train_data.n_token  # size of vocabulary
model = TransformerModel(n_token)

train_model(model, train_data, 10)
save_model(model)

test_loss = evaluate_model(model, test_data)
print('perplexity on test data: {:.2f}'.format(math.exp(test_loss)))