import os
from argparse import ArgumentParser

import torch
from tqdm import tqdm
from gensim.models import KeyedVectors

from nn.modeling import MLP
from nn.data_loader import PNDataLoader
from nn.utils import TEST_FILE, W2V_MODEL_FILE
from nn.utils import metric_fn, loss_fn


def main():
    # Training settings
    parser = ArgumentParser()
    parser.add_argument('-d', '--device', default=None, type=str,
                        help='indices of GPUs to enable (default: None)')
    parser.add_argument('-b', '--batch-size', type=int, default=256,
                        help='number of batch size for training')
    parser.add_argument('--load-path', type=str, default='result/model.pth',
                        help='path to trained model')
    parser.add_argument('--env', choices=['local', 'server'], default='server',
                        help='development environment')
    args = parser.parse_args()

    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    device = torch.device('cuda:0' if torch.cuda.is_available() and args.device is not None else 'cpu')

    # setup data_loader instances
    model_w2v = KeyedVectors.load_word2vec_format(W2V_MODEL_FILE[args.env], binary=True)
    test_data_loader = PNDataLoader(TEST_FILE[args.env], model_w2v, args.batch_size, shuffle=False, num_workers=2)

    # build model architecture
    model = MLP(word_dim=128, hidden_dim=100)
    # load state dict
    state_dict = torch.load(args.load_path, map_location=device)
    model.load_state_dict(state_dict)

    model.to(device)

    # test
    model.eval()
    with torch.no_grad():
        total_loss = 0
        total_correct = 0
        for batch_idx, (source, mask, target) in tqdm(enumerate(test_data_loader)):
            source = source.to(device)  # (b, len, dim)
            mask = mask.to(device)      # (b, len)
            target = target.to(device)  # (b)

            output = model(source, mask)  # (b, 2)

            total_loss += loss_fn(output, target)
            total_correct += metric_fn(output, target)
    print(f'test_loss={total_loss / test_data_loader.n_samples:.3f}', end=' ')
    print(f'test_accuracy={total_correct / test_data_loader.n_samples:.3f}\n')


if __name__ == '__main__':
    main()