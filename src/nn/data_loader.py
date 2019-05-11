from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from constants import PAD


class PosNegDataset(Dataset):
    def __init__(self,
                 path: str,
                 word2id: Dict[str, int],
                 wlim: Optional[int]):
        self.word2id = word2id
        self.wlim = wlim
        self.sources, self.targets = self._load(path)
        self.max_phrase_len: int = max(len(phrase) for phrase in self.sources)

    def __len__(self) -> int:
        return len(self.sources)

    def __getitem__(self, idx) -> Tuple[List[int], List[int], int]:
        source = self.sources[idx]  # (len)
        mask = [1] * len(source)    # (len)
        target = self.targets[idx]  # ()
        return source, mask, target

    def _load(self, path: str) -> Tuple[List[List[int]], List[int]]:
        sources, targets = [], []
        with open(path) as f:
            for line in f:
                tag, body = line.strip().split('\t')
                assert tag in ['1', '-1']
                targets.append(int(tag == '1'))
                ids: List[int] = []
                for mrph in body.split():
                    if mrph in self.word2id:
                        ids.append(self.word2id[mrph])
                    else:
                        ids.append(self.word2id['<UNK>'])
                if self.wlim is not None and len(ids) > self.wlim:
                    ids = ids[-self.wlim:]  # limit word length
                sources.append(ids)
        assert len(sources) == len(targets)
        return sources, targets


class PosNegDataLoader(DataLoader):
    def __init__(self,
                 path: str,
                 word2id: Dict[str, int],
                 wlim: Optional[int],
                 batch_size: int,
                 shuffle: bool,
                 num_workers: int):
        self.dataset = PosNegDataset(path, word2id, wlim)
        self.n_samples = len(self.dataset)
        super(PosNegDataLoader, self).__init__(self.dataset,
                                               batch_size=batch_size,
                                               shuffle=shuffle,
                                               num_workers=num_workers,
                                               collate_fn=my_collate_fn)


def my_collate_fn(batch: List[Tuple[List[int], List[int], int]]
                  ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sources, masks, targets = [], [], []
    max_seq_in_batch = max(len(sample[0]) for sample in batch)
    for sample in batch:
        source, mask, target = sample
        pad = [PAD] * (max_seq_in_batch - len(source))
        sources.append(source+pad)
        masks.append(mask+pad)
        targets.append(target)
    return torch.LongTensor(sources), torch.LongTensor(masks), torch.LongTensor(targets)
