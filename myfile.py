from data_provider import LMData, LibriSpeechRawIter
from model import TransformerModel, train_model, evaluate_model, save_model

train_iter = LibriSpeechRawIter(data_mode='train')
# test_iter = LibriSpeechRawIter(data_mode='test')
train_data = LMData(train_iter, train_iter, 20, 35)
# test_data = LMData(train_iter, raw_text_iter_test, 10, 35)

print(train_data.n_token)
# print(test_data.n_token)
n_token = train_data.n_token  # size of vocabulary
model = TransformerModel(n_token)

train_model(model, train_data, 12)
save_model(model)

# test_loss = evaluate_model(model, test_data)
# print(test_loss)
