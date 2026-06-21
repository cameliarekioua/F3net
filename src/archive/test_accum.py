import os
import random
import torch
import numpy as np

from utils import FFDataset, get_dataset
from trainer_v3 import Trainer

DATASET_PATH    = os.path.expanduser('~/F3Net/data/FF++_preprocessed')
PRETRAINED_PATH = os.path.expanduser('~/F3Net/checkpoints/xception-b5690688.pth')

BATCH_SIZE  = 12
FRAME_NUM   = 270
ACCUM_STEPS = 10
TEST_STEPS  = 50  # nombre de micro-steps à tester

GPU_IDS = [0]


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    set_seed()

    dataset_real = FFDataset(
        dataset_root=os.path.join(DATASET_PATH, 'train', 'real'),
        size=299, frame_num=FRAME_NUM, augment=True
    )
    dataloader_real = torch.utils.data.DataLoader(
        dataset_real, batch_size=BATCH_SIZE // 2,
        shuffle=True, num_workers=2
    )

    dataset_fake, _ = get_dataset(
        name='train', size=299, root=DATASET_PATH,
        frame_num=FRAME_NUM, augment=True
    )
    dataloader_fake = torch.utils.data.DataLoader(
        dataset_fake, batch_size=BATCH_SIZE // 2,
        shuffle=True, num_workers=2
    )

    model = Trainer(GPU_IDS, PRETRAINED_PATH)
    model.model.train()
    model.total_steps = 0

    real_iter = iter(dataloader_real)
    fake_iter = iter(dataloader_fake)

    print(f"{'step':>5} | {'loss':>8} | {'grad_norm':>10} | step() appelé ?")
    print("-" * 50)

    for step in range(1, TEST_STEPS + 1):
        data_real = next(real_iter)
        data_fake = next(fake_iter)

        if data_real.shape[0] != data_fake.shape[0]:
            continue

        bz    = data_real.shape[0]
        data  = torch.cat([data_real, data_fake], dim=0)
        label = torch.cat([
            torch.zeros(bz, dtype=torch.long),
            torch.ones(bz,  dtype=torch.long)
        ], dim=0)

        idx   = list(range(data.shape[0]))
        random.shuffle(idx)
        data  = data[idx]
        label = label[idx]

        model.set_input(data, label)
        loss = model.optimize_weight(accumulation_steps=ACCUM_STEPS)
        model.total_steps += 1

        # Vérifie la norme des gradients accumulés (avant step/zero_grad)
        total_norm = 0.0
        for p in model.model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        total_norm = total_norm ** 0.5

        did_step = (model.total_steps % ACCUM_STEPS == 0)
        if did_step:
            model.step()

        print(f"{step:5d} | {loss:8.4f} | {total_norm:10.4f} | {did_step}")

    print("\nTest terminé sans erreur.")


if __name__ == '__main__':
    main()
