import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm

from handler import ReturnDataset, VolatilityDataset, CorrelationDataset
from helper import calc_accuracy
from model import RankNet

epochs = 300
batch_size = 128
device = 'cuda:0'
log_dir = './runs/tw50/ranknet'

# Dataset
train_pkl_file = {'price': './Data/pct_returns_train.npy',
                  'volume': './Data/pct_volume_train.npy'}
train_dataset_q1 = ReturnDataset(0, train_pkl_file, formation=1350, holding=270)
train_dataset_q2 = VolatilityDataset(1, train_pkl_file, formation=1350, holding=270)
train_dataset_q3 = CorrelationDataset(2, train_pkl_file, formation=1350, holding=270)
train_dataset = ConcatDataset((train_dataset_q1, train_dataset_q2, train_dataset_q3))
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)

test_pkl_file = {'price': './Data/pct_returns_test.npy',
                 'volume': './Data/pct_volume_test.npy'}
test_dataset_q1 = ReturnDataset(0, test_pkl_file, formation=1350, holding=270)
test_dataset_q2 = VolatilityDataset(1, test_pkl_file, formation=1350, holding=270)
test_dataset_q3 = CorrelationDataset(2, test_pkl_file, formation=1350, holding=270)
test_dataset = ConcatDataset((test_dataset_q1, test_dataset_q2, test_dataset_q3))
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)

num_q = len(train_dataset.datasets)

# Model
ranknet = RankNet(num_q).to(device)

# Optimizer
optimizer = torch.optim.Adam(ranknet.parameters())

# Loss
criterion = nn.BCEWithLogitsLoss(reduction='none')

# Tensorboard
train_writer = SummaryWriter(log_dir)
test_writer = SummaryWriter(log_dir)

for e in range(epochs):
    # Training phase
    ranknet.train()
    with tqdm(total=len(train_dataloader), ncols=130) as progress_bar:
        for b, (x1, x2, q, a) in enumerate(train_dataloader):
            x1 = x1.to(device)
            x2 = x2.to(device)
            q = q.to(device)
            a = a.to(device)

            pred = ranknet(x1, x2, q)
            loss = criterion(pred, a)

            optimizer.zero_grad()
            loss.mean().backward()
            optimizer.step()

            loss_summary = {f'q{i}': loss[q == i].mean() for i in range(num_q)}
            acc_summary = {f'q{i}': calc_accuracy(pred[q == i], a[q == i], mode='binary') for i in range(num_q)}
            train_writer.add_scalars('train_loss', loss_summary, global_step=e)
            train_writer.add_scalars('train_acc', acc_summary, global_step=e)

            progress_bar.update(1)

    # Testing phase
    ranknet.eval()
    with tqdm(total=len(test_dataloader), ncols=130) as progress_bar:
        with torch.no_grad():
            for b, (x1, x2, q, a) in enumerate(test_dataloader):
                x1 = x1.to(device)
                x2 = x2.to(device)
                q = q.to(device)
                a = a.to(device)

                pred = ranknet(x1, x2, q)
                loss = criterion(pred, a)

                loss_summary = {f'q{i}': loss[q == i].mean() for i in range(num_q)}
                acc_summary = {f'q{i}': calc_accuracy(pred[q == i], a[q == i], mode='binary') for i in range(num_q)}
                test_writer.add_scalars('test_loss', loss_summary, global_step=e)
                test_writer.add_scalars('test_acc', acc_summary, global_step=e)

                progress_bar.update(1)

train_writer.close()
test_writer.close()
