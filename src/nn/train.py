import os
from argparse import ArgumentParser

from tqdm import tqdm

import torch

from nn.modeling import MLP
from nn.data_loader import PNDataLoader
from nn.utils import TRAIN_FILE, VALID_FILE
from nn.utils import metric_fn, loss_fn


def main():
    # Training settings
    parser = ArgumentParser()
    parser.add_argument('-d', '--device', default=None, type=str,
                        help='indices of GPUs to enable (default: None)')
    parser.add_argument('-b', '--batch-size', type=int, default=256,
                        help='number of batch size for training')
    parser.add_argument('-e', '--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--save-dir', type=str, default='result/',
                        help='directory where trained model is saved')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='learning rate')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 1)')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    device = torch.device('cuda:0' if torch.cuda.is_available() and args.device is not None else 'cpu')

    # setup data_loader instances
    train_data_loader = PNDataLoader(TRAIN_FILE, args.batch_size, shuffle=True, num_workers=2)
    valid_data_loader = PNDataLoader(VALID_FILE, args.batch_size, shuffle=False, num_workers=2)

    # build model architecture
    model = MLP(word_dim=128, hidden_dim=100)
    model.to(device)

    # build optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_acc = 0

    for epoch in range(args.epochs):
        print(f'*** epoch {epoch} ***')
        # train
        model.train()
        total_loss = 0
        total_correct = 0
        for batch_idx, (source, mask, target) in tqdm(enumerate(train_data_loader)):
            source = source.to(device)  # (b, len, dim)
            mask = mask.to(device)      # (b, len)
            target = target.to(device)  # (b)
            optimizer.zero_grad()

            output = model(source, mask)  # (b, 2)

            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_correct += metric_fn(output, target)
        print(f'train_loss={total_loss / train_data_loader.n_samples:.3f}', end=' ')
        print(f'train_accuracy={total_correct / train_data_loader.n_samples:.3f}')

        # validation
        model.eval()
        with torch.no_grad():
            total_loss = 0
            total_correct = 0
            for batch_idx, (source, mask, target) in tqdm(enumerate(valid_data_loader)):
                source = source.to(device)  # (b, len, dim)
                mask = mask.to(device)      # (b, len)
                target = target.to(device)  # (b)

                output = model(source, mask)  # (b, 2)

                total_loss += loss_fn(output, target)
                total_correct += metric_fn(output, target)
        valid_acc = total_correct / valid_data_loader.n_samples
        print(f'valid_loss={total_loss / valid_data_loader.n_samples:.3f}', end=' ')
        print(f'valid_accuracy={valid_acc:.3f}\n')
        if valid_acc > best_acc:
            torch.save(model.state_dict(), os.path.join(args.save_dir, 'model.pth'))


if __name__ == '__main__':
    main()
