# F³-Net — Reproduction & Deepfake Detection Demo

Reproduction of **"Thinking in Frequency: Face Forgery Detection by Mining Frequency-aware Clues"** (Qian et al., ECCV 2020) as part of an AI project at Télécom Paris.

---

## What this repo does

This project reproduces the F³-Net architecture for deepfake detection on the FaceForensics++ dataset (c40 compression level).

F³-Net detects deepfakes by analyzing **frequency-domain artifacts** that remain detectable even after heavy video compression. The model combines:
- **FAD** (Frequency-Aware Decomposition): global spectral filtering with learnable frequency bands
- **LFS** (Local Frequency Statistics): patch-level DCT statistics over sliding windows
- **MixBlock**: cross-attention fusion between the two branches at mid-network (blocks 7 and 12 of Xception)

## Setup

**Requirements:** Python 3.8+, PyTorch 1.10+, CUDA

```bash
git clone https://github.com/YOUR_USERNAME/F3Net.git
cd F3Net
python -m venv env
source env/bin/activate
pip install torch torchvision facenet-pytorch scikit-learn tqdm opencv-python-headless gradio
```

Download the Xception pretrained weights and place them in `checkpoints/`:
```bash
mkdir -p checkpoints
wget https://huggingface.co/spaces/asdasdasdasd/Face-forgery-detection/resolve/main/xception-b5690688.pth -P checkpoints/
```

---

## Data preprocessing

Download FaceForensics++ following the [official instructions](https://github.com/ondyari/FaceForensics). Then run:

```bash
cd src
python preprocess_ff.py
```

This extracts 270 frames per video, detects faces with MTCNN, applies a ×1.3 conservative crop around each face (as specified in the original paper), and saves 299×299 JPEG patches organized by split and manipulation method. Expected output: ~1.35M face patches across all splits.

---

## Training

```bash
cd src
python train.py
```

Key hyperparameters (top of `train.py`):
```python
DATASET_PATH   = '~/F3Net/data/FF++_preprocessed'
PRETRAINED_PATH = '~/F3Net/checkpoints/xception-b5690688.pth'
BATCH_SIZE     = 12
MAX_EPOCHS     = 5
```

Best checkpoint is saved automatically based on validation AUC.

---

## Evaluation with optimal threshold

```bash
cd src
python - << 'EOF'
import torch, numpy as np
from sklearn.metrics import roc_curve, auc, accuracy_score
from torch.utils.data import DataLoader
from utils import FFDataset, get_dataset
from trainer import Trainer

PRETRAINED_PATH = '../checkpoints/xception-b5690688.pth'
DATASET_PATH    = '../data/FF++_preprocessed'
CKPT_PATH       = '../experiments/f3net_c40/best.pth'

model = Trainer([0], PRETRAINED_PATH)
model.load(CKPT_PATH)
model.model.eval()

def get_scores(model, dataset_path, mode):
    dataset_real = FFDataset(f'{dataset_path}/{mode}/real', size=299, frame_num=50, augment=False)
    dataset_fake, _ = get_dataset(name=mode, root=dataset_path, size=299, frame_num=50, augment=False)
    y_true, y_pred = [], []
    with torch.no_grad():
        for i, subset in enumerate([dataset_real] + list(dataset_fake.datasets)):
            for imgs in DataLoader(subset, batch_size=64, num_workers=4):
                logit = model.forward(imgs.cuda())
                probs = torch.softmax(logit, dim=1)[:, 1]
                y_pred.extend(probs.cpu().tolist())
                y_true.extend([0 if i == 0 else 1] * imgs.size(0))
    return np.array(y_true), np.array(y_pred)

y_val, p_val = get_scores(model, DATASET_PATH, 'valid')
best_acc, best_t = max((accuracy_score(y_val, p_val > t), t) for t in np.arange(0, 1, 0.01))
print(f'Optimal threshold on validation: {best_t:.2f}  (Acc: {best_acc:.4f})')

y_test, p_test = get_scores(model, DATASET_PATH, 'test')
fpr, tpr, _ = roc_curve(y_test, p_test)
print(f'Test AUC: {auc(fpr, tpr):.4f}')
print(f'Test Acc (θ={best_t:.2f}): {accuracy_score(y_test, p_test > best_t):.4f}')
EOF
```

---

## References

```bibtex
@inproceedings{qian2020thinking,
  title     = {Thinking in Frequency: Face Forgery Detection by Mining Frequency-aware Clues},
  author    = {Qian, Yuyang and Yin, Guojun and Sheng, Lu and Chen, Zixuan and Shao, Jing},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2020}
}
```

Model implementation based on:
- [Leminhbinh0209/F3Net](https://github.com/Leminhbinh0209/F3Net) — reference architecture with MixBlock
- [yyk-wew/F3Net](https://github.com/yyk-wew/F3Net) — training pipeline

---

*Télécom Paris — Projet Intégrateur IA — 2026*
