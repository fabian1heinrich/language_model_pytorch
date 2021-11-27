import torch
import time

from torch import nn
from datetime import datetime

from torch.optim import optimizer

from data_provider import generate_square_subsequent_mask, seq2str


def train_model(
    model: nn.Module,
    train_data,
    epochs: int = 5,
    lr: float = 5.0,
):

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

    src_mask = generate_square_subsequent_mask(train_data.seq_len)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    print('>> training started@{} <<'.format(timestamp))
    elapsed_time = time.time()
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        batch_wait, model_train = [], []
        time1, time2, time3 = 0, 0, 0
        for inputs, targets, index in train_data:

            time1 = time.time()
            optimizer.zero_grad()

            outputs = model(inputs, src_mask)
            loss = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            time2 = time.time()

            train_loss += loss.item()
            batch_wait.append(time1 - time3)
            model_train.append(time2 - time1)
            time3 = time.time()

        train_loss /= len(train_data)
        batch_wait_mean = sum(batch_wait[1:]) / (len(batch_wait) - 1) * 10**3
        model_train_mean = sum(model_train) / len(model_train) * 10**3
        scheduler.step()
        print(
            'epoch #{}/{} | loss: {:.2f} | elapsed time: {:>3.0f} s | batch_wait: {:.2f} ms | model_train: {:.2f} ms'
            .format(
                epoch + 1,
                epochs,
                train_loss,
                time.time() - elapsed_time,
                batch_wait_mean,
                model_train_mean,
            ))

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    print('>> training finished@{} <<'.format(timestamp))