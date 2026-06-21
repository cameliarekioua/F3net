import torch
import torch.nn as nn
from f3net import F3Net


def init_model(model, gpu_ids):
    model = model.to(f'cuda:{gpu_ids[0]}')
    model = nn.DataParallel(model, gpu_ids)
    return model


class Trainer:
    def __init__(self, gpu_ids, pretrained_path):
        self.device = torch.device(f'cuda:{gpu_ids[0]}') if gpu_ids else torch.device('cpu')
        self.model  = F3Net(pretrained_path=pretrained_path)
        self.model  = init_model(self.model, gpu_ids)
        self.loss_fn = nn.CrossEntropyLoss()

        # SGD conforme à l'article F3Net (ECCV 2020)
        self.optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=0.002,
            momentum=0.9,
            weight_decay=0
        )

    def set_input(self, data, label):
        self.data  = data.to(self.device)
        self.label = label.to(self.device)

    def forward(self, x):
        _, logit = self.model(x)
        return logit

    def optimize_weight(self):
        _, logit = self.model(self.data)
        loss = self.loss_fn(logit, self.label)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        state_dict = torch.load(path)
        self.model.load_state_dict(state_dict)
