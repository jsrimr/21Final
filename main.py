import torch
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import os, nltk
import numpy as np

from miscc.config import cfg, cfg_from_file
import pprint
import datetime
import dateutil.tz

from utils.data_utils import CUBDataset
from utils.trainer import trainer

# Set a config file as 'train_birds.yml' in training, as 'eval_birds.yml' for evaluation
cfg_from_file('cfg/train_birds.yml') # eval_birds.yml

os.environ['CUDA_VISIBLE_DEVICES'] = cfg.GPU_ID

now = datetime.datetime.now(dateutil.tz.tzlocal())
timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
output_dir = 'sample/%s_%s_%s' % (cfg.DATASET_NAME, cfg.CONFIG_NAME, timestamp)

imsize = cfg.TREE.BASE_SIZE * (4 ** (cfg.TREE.BRANCH_NUM - 1))
image_transform = transforms.Compose([
    transforms.Resize(int(imsize)),
    transforms.RandomCrop(imsize),
    transforms.RandomHorizontalFlip()])

train_dataset = CUBDataset(cfg.DATA_DIR, transform=image_transform, split='train')
test_dataset = CUBDataset(cfg.DATA_DIR, transform=image_transform, split='test')

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE,
        drop_last=True, shuffle=True, num_workers=int(cfg.WORKERS))
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=cfg.BATCH_SIZE,
        drop_last=True, shuffle=False, num_workers=int(cfg.WORKERS))


if __name__ == "__main__":
    # algo = trainer(output_dir, train_dataset, train_dataloader, test_dataset, test_dataloader)
    algo = trainer(output_dir, train_dataloader, train_dataset.n_words, train_dataset.ixtoword, train_dataset)

    if cfg.TRAIN.FLAG:
        algo.train()
    else:
        algo.generate_eval_data()