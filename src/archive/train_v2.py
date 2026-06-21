import os
import random
import torch
import numpy as np

from utils import FFDataset, get_dataset, evaluate, setup_logger
from trainer_v2 import Trainer


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATASET_PATH    = os.path.expanduser('~/F3Net/data/FF++_preprocessed')
PRETRAINED_PATH = os.path.expanduser('~/F3Net/checkpoints/xception-b5690688.pth')
EXPERIMENT_DIR  = os.path.expanduser('~/F3Net/experiments/f3net_c40_v2_fixed')

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
BATCH_SIZE  = 12
MAX_EPOCHS  = 5
FRAME_NUM   = 270
LOG_FREQ    = 40
EVAL_FREQ   = 10
GPU_IDS     = [0]


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    set_seed()
    os.makedirs(EXPERIMENT_DIR, exist_ok=True)
    logger = setup_logger(EXPERIMENT_DIR, 'train.log', 'train_logger')

    logger.info(f'Dataset   : {DATASET_PATH}')
    logger.info(f'Checkpoint: {PRETRAINED_PATH}')
    logger.info(f'Batch size: {BATCH_SIZE}  |  Epochs: {MAX_EPOCHS}')

    # ------------------------------------------------------------------
    # Dataloaders
    # ------------------------------------------------------------------
    dataset_real = FFDataset(
        dataset_root=os.path.join(DATASET_PATH, 'train', 'real'),
        size=299, frame_num=FRAME_NUM, augment=True
    )
    dataloader_real = torch.utils.data.DataLoader(
        dataset_real, batch_size=BATCH_SIZE // 2,
        shuffle=True, num_workers=4, drop_last=True
    )

    dataset_fake, _ = get_dataset(
        name='train', size=299, root=DATASET_PATH,
        frame_num=FRAME_NUM, augment=True
    )
    dataloader_fake = torch.utils.data.DataLoader(
        dataset_fake, batch_size=BATCH_SIZE // 2,
        shuffle=True, num_workers=4, drop_last=True
    )

    steps_per_epoch = len(dataloader_real)
    eval_every      = max(1, steps_per_epoch // EVAL_FREQ)

    # T_max = nombre réel de steps sur toute la durée d'entraînement.
    # C'est la correction clé par rapport à v2 (ancien T_max=150000
    # fixe qui ne correspondait pas au nombre réel de steps et
    # provoquait une remontée du lr en fin d'entraînement).
    total_steps = steps_per_epoch * MAX_EPOCHS

    logger.info(f'Steps per epoch: {steps_per_epoch}')
    logger.info(f'Total steps    : {total_steps}  (T_max du scheduler)')
    logger.info(f'Optimizer      : SGD  lr=0.002  momentum=0.9')
    logger.info(f'Scheduler      : CosineAnnealingLR  T_max={total_steps}  eta_min=1e-6')

    # ------------------------------------------------------------------
    # Modèle
    # ------------------------------------------------------------------
    model = Trainer(GPU_IDS, PRETRAINED_PATH)
    model.total_steps = 0
    best_auc = 0.0

    # Cosine annealing sur le nombre RÉEL de steps
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        model.optimizer,
        T_max=total_steps,
        eta_min=1e-6
    )

    # ------------------------------------------------------------------
    # Boucle d'entraînement
    # ------------------------------------------------------------------
    for epoch in range(MAX_EPOCHS):
        logger.info(f'--- Epoch {epoch} ---')
        model.model.train()

        real_iter = iter(dataloader_real)
        fake_iter = iter(dataloader_fake)

        step = 0
        while step < steps_per_epoch:
            try:
                data_real = next(real_iter)
                data_fake = next(fake_iter)
            except StopIteration:
                break

            if data_real.shape[0] != data_fake.shape[0]:
                continue

            bz    = data_real.shape[0]
            data  = torch.cat([data_real, data_fake], dim=0)
            label = torch.cat([
                torch.zeros(bz, dtype=torch.long),
                torch.ones(bz,  dtype=torch.long)
            ], dim=0)

            idx = list(range(data.shape[0]))
            random.shuffle(idx)
            data  = data[idx]
            label = label[idx]

            model.set_input(data, label)
            loss = model.optimize_weight()

            # Scheduler step à chaque micro-step
            scheduler.step()

            step              += 1
            model.total_steps += 1

            if model.total_steps % LOG_FREQ == 0:
                lr_current = scheduler.get_last_lr()[0]
                logger.info(
                    f'  step {model.total_steps:5d}  '
                    f'loss: {loss:.4f}  lr: {lr_current:.6f}'
                )

            if step % eval_every == 0:
                model.model.eval()
                val_auc, val_r, val_f = evaluate(model, DATASET_PATH, mode='valid')
                logger.info(
                    f'  [Val  @ epoch {epoch} step {step}] '
                    f'AUC: {val_auc:.4f}  r_acc: {val_r:.4f}  f_acc: {val_f:.4f}'
                )
                if val_auc > best_auc:
                    best_auc = val_auc
                    ckpt_path = os.path.join(EXPERIMENT_DIR, 'best.pth')
                    model.save(ckpt_path)
                    logger.info(f'  -> New best AUC {best_auc:.4f} — model saved.')
                model.model.train()

    # ------------------------------------------------------------------
    # Évaluation finale
    # ------------------------------------------------------------------
    model.model.eval()
    test_auc, test_r, test_f = evaluate(model, DATASET_PATH, mode='test')
    logger.info(
        f'[Final Test] AUC: {test_auc:.4f}  r_acc: {test_r:.4f}  f_acc: {test_f:.4f}'
    )


if __name__ == '__main__':
    main()
