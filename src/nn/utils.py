import torch
import torch.nn.functional as F


TRAIN_FILE = '/Users/NobuhiroUeda/PycharmProjects/2019_shinjin_baseline3/data/train/samples.txt'
VALID_FILE = '/Users/NobuhiroUeda/PycharmProjects/2019_shinjin_baseline3/data/valid/samples.txt'
TEST_FILE = '/Users/NobuhiroUeda/PycharmProjects/2019_shinjin_baseline3/data/test/samples.txt'
W2V_MODEL_FILE = '/Users/NobuhiroUeda/PycharmProjects/2019_shinjin_baseline3/data/w2v.midasi.128.100K.bin'


def loss_fn(output: torch.Tensor,  # (b, 2)
            target: torch.Tensor   # (b)
            ) -> torch.Tensor:     # ()
    softmax = F.softmax(output, dim=1)  # (b, 2)
    loss = F.binary_cross_entropy(softmax[:, 1], target.float(), reduction='sum')
    return loss


def metric_fn(output: torch.Tensor,  # (b, 2)
              target: torch.Tensor   # (b)
              ) -> int:
    prediction = torch.argmax(output, dim=1)
    return (prediction == target).sum().item()
