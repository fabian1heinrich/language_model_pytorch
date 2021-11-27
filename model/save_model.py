from genericpath import exists
import os
import random
import string
import torch
from datetime import datetime


def save_model(model, path='saved_models/'):

    if not os.path.exists(path):
        os.mkdir(path)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_')
    random_string = ''.join(random.choices(string.ascii_uppercase, k=7))
    name = timestamp + random_string
    save_path = os.path.join(path, name)

    torch.save(
        {
            'model_state_dict': model.state_dict(),
        },
        save_path,
    )
    print('{} saved'.format(save_path))