import numpy as np
import torch
import torch.nn.functional as F
from gensim.models import KeyedVectors

from constants import PAD, UNK


TRAIN_FILE = {'local': '/Users/NobuhiroUeda/PycharmProjects/2019_shinjin_baseline3/data/train.txt',
              'server': '/mnt/hinoki/ueda/shinjin2019/acp-2.0/train.txt'}
VALID_FILE = {'local': '/Users/NobuhiroUeda/PycharmProjects/2019_shinjin_baseline3/data/valid.txt',
              'server': '/mnt/hinoki/ueda/shinjin2019/acp-2.0/valid.txt'}
TEST_FILE = {'local': '/Users/NobuhiroUeda/PycharmProjects/2019_shinjin_baseline3/data/test.txt',
             'server': '/mnt/hinoki/ueda/shinjin2019/acp-2.0/test.txt'}
W2V_MODEL_FILE = {'local': '/Users/NobuhiroUeda/PycharmProjects/2019_shinjin_baseline3/data/w2v.midasi.128.100K.bin',
                  'server': '/mnt/windroot/share/word2vec/2016.08.02/w2v.midasi.128.100K.bin'}


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


# use vocab of model_w2v
def word2id(model_w2v):
    word_to_id = {word: word_id + 1 for word_id, word in enumerate(model_w2v.vocab.keys())}
    word_to_id['<PAD>'] = PAD
    return word_to_id


def load_word_embedding(model_w2v: KeyedVectors) -> np.ndarray:
    embedding = np.zeros((len(model_w2v.vocab) + 1, model_w2v.vector_size), 'f')  # (vocab_size, dim)
    embedding[1:] = model_w2v.vectors
    return embedding
