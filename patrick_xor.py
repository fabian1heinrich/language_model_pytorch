import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader

import torch
import torch.nn as nn

from torch.utils.data import Dataset


class XORData(Dataset):
    def __init__(self, length, batch_size):
        super().__init__()

        self.length = length
        self.batch_size = batch_size

    def __getitem__(self, index):
        inputs = torch.randint(
            low=0,
            high=2,
            size=(self.batch_size, 2),
        )
        targets = torch.bitwise_xor(inputs[:, 0], inputs[:, 1])

        return inputs, targets.unsqueeze_(dim=1)

    def __len__(self):
        return self.length // self.batch_size


model = nn.Sequential(
    nn.Linear(2, 3),
    nn.Sigmoid(),
    nn.Linear(3, 1),
)

data = XORData(10000, 10)
data_loader = DataLoader(data, batch_size=1)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

for epoch in range(12):
    train_loss = 0
    for batch in data_loader:
        input, target = batch

        optimizer.zero_grad()
        output = model(input.float())
        loss = criterion(output, target.float())
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    print('epoch: {} | loss: {:.2f}'.format(
        epoch + 1,
        train_loss / len(data_loader),
    ))

print(model.layer[0].weight)