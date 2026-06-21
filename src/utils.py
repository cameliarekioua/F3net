import os
import sys
import random
import logging
import numpy as np
import torch
from torch.utils import data
from torchvision import transforms as trans
from sklearn.metrics import accuracy_score, roc_curve
from sklearn.metrics import auc as cal_auc
from PIL import Image


class FFDataset(data.Dataset):

    def __init__(self, dataset_root, frame_num=300, size=299, augment=True):
        self.data_root  = dataset_root
        self.frame_num  = frame_num
        self.train_list = self._collect_images(self.data_root)
        if augment:
            self.transform = trans.Compose([trans.RandomHorizontalFlip(p=0.5), trans.ToTensor()])
        else:
            self.transform = trans.ToTensor()
        self.max_val = 1.
        self.min_val = -1.
        self.size = size

    def _collect_images(self, root):
        image_path_list = []
        for split in os.listdir(root):
            split_root = os.path.join(root, split)
            if not os.path.isdir(split_root):
                continue
            img_list = os.listdir(split_root)
            random.shuffle(img_list)
            img_list = img_list[:self.frame_num]
            for img in img_list:
                image_path_list.append(os.path.join(split_root, img))
        return image_path_list

    def __getitem__(self, index):
        img = Image.open(self.train_list[index]).convert('RGB')
        img = img.resize((self.size, self.size))
        img = self.transform(img)
        img = img * (self.max_val - self.min_val) + self.min_val
        return img

    def __len__(self):
        return len(self.train_list)


def get_dataset(name='train', size=299, root='', frame_num=300, augment=True):
    fake_root = os.path.join(root, name, 'fake')
    fake_list = ['Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures']
    datasets  = []
    for method in fake_list:
        path = os.path.join(fake_root, method)
        dset = FFDataset(path, frame_num, size, augment)
        datasets.append(dset)
    return torch.utils.data.ConcatDataset(datasets), len(fake_list)


def evaluate(model, data_path, mode='valid'):
    real_root    = os.path.join(data_path, mode, 'real')
    dataset_real = FFDataset(dataset_root=real_root, size=299, frame_num=50, augment=False)
    dataset_fake, _ = get_dataset(name=mode, root=data_path, size=299, frame_num=50, augment=False)
    full_dataset = torch.utils.data.ConcatDataset([dataset_real, dataset_fake])

    y_true, y_pred = [], []
    with torch.no_grad():
        for i, subset in enumerate(full_dataset.datasets):
            loader = torch.utils.data.DataLoader(subset, batch_size=64, shuffle=False, num_workers=4)
            for imgs in loader:
                label = torch.zeros(imgs.size(0)) if i == 0 else torch.ones(imgs.size(0))
                imgs  = imgs.cuda()
                logit = model.forward(imgs)
                probs = torch.softmax(logit, dim=1)[:, 1]
                y_pred.extend(probs.cpu().tolist())
                y_true.extend(label.tolist())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    fpr, tpr, _ = roc_curve(y_true, y_pred, pos_label=1)
    auc = cal_auc(fpr, tpr)

    r_acc = accuracy_score(y_true[y_true == 0], (y_pred[y_true == 0] > 0.5).astype(int))
    f_acc = accuracy_score(y_true[y_true == 1], (y_pred[y_true == 1] > 0.5).astype(int))

    return auc, r_acc, f_acc


def setup_logger(work_dir, logfile_name='log.txt', logger_name='logger'):
    logger = logging.getLogger(logger_name)
    if logger.hasHandlers():
        return logger

    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s")

    sh = logging.StreamHandler(stream=sys.stdout)
    sh.setLevel(logging.DEBUG)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    os.makedirs(work_dir, exist_ok=True)
    fh = logging.FileHandler(os.path.join(work_dir, logfile_name))
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger
