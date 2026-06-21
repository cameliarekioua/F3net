import torch
import torch.nn as nn
from f3net import F3Net


def init_model(model, gpu_ids):
    model = model.to(f'cuda:{gpu_ids[0]}')
    model = nn.DataParallel(model, gpu_ids)
    return model


class Trainer:
    def __init__(self, gpu_ids, pretrained_path, accum_steps=10):
        self.device      = torch.device(f'cuda:{gpu_ids[0]}') if gpu_ids else torch.device('cpu')
        self.accum_steps = accum_steps
        self.model       = F3Net(pretrained_path=pretrained_path)
        self.model       = init_model(self.model, gpu_ids)
        self.loss_fn     = nn.CrossEntropyLoss()

        # Linear scaling rule : lr de base 0.0002 (batch 12)
        # multiplié par accum_steps pour un batch effectif de 12*10=120.
        # Donne lr=0.002, identique au lr SGD de l'article pour batch 128.
        lr_scaled = 0.0002 * accum_steps
        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=lr_scaled,
            betas=(0.9, 0.999)
        )

        # Compteur interne pour savoir combien de micro-steps ont été
        # accumulés depuis le dernier optimizer.step()
        self._accum_count = 0

        # zero_grad initial pour l'accumulation
        self.optimizer.zero_grad()

    def set_input(self, data, label):
        self.data  = data.to(self.device)
        self.label = label.to(self.device)

    def forward(self, x):
        _, logit = self.model(x)
        return logit

    def optimize_weight(self):
        """
        Accumule les gradients sur accum_steps micro-batchs,
        puis applique un optimizer.step().
        Retourne (loss_value, did_step) :
          - loss_value : loss du micro-batch courant (non normalisée, pour les logs)
          - did_step   : True si un vrai step d'optimisation a eu lieu
        """
        _, logit = self.model(self.data)
        loss = self.loss_fn(logit, self.label)

        # Normalisation : la somme des gradients accumulés divisée par
        # accum_steps est équivalente à la moyenne sur le batch effectif.
        (loss / self.accum_steps).backward()
        self._accum_count += 1

        did_step = False
        if self._accum_count >= self.accum_steps:
            self.optimizer.step()
            self.optimizer.zero_grad()
            self._accum_count = 0
            did_step = True

        return loss.item(), did_step

    def flush(self):
        """
        Applique les gradients résiduels en fin d'epoch si
        _accum_count > 0 (batch incomplet).
        """
        if self._accum_count > 0:
            self.optimizer.step()
            self.optimizer.zero_grad()
            self._accum_count = 0

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        state_dict = torch.load(path)
        self.model.load_state_dict(state_dict)
