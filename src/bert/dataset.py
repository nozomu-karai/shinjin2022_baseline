from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader

class PosNegDataset(Dataset):
    def __init__(self,
                 path: str,
                 tokenizer,
                 max_seq_len: int):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.inputs, self.labels = self._load(path)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx) -> Dict:
        outputs = self.tokenizer(self.inputs[idx], truncation=True, max_length = self.max_seq_len,
                                 padding='max_length', return_tensors='pt')
        input_ids = outputs['input_ids']
        token_type_ids = outputs['token_type_ids']
        attention_mask = outputs['attention_mask']

        return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'token_type_ids': token_type_ids,
        'label': self.labels[idx]
        }

    def _load(self, path: str) -> Tuple[List[str], List[int]]:
        inputs, labels = [], []
        with open(path) as f:
            for line in f:
                tag, body = line.strip().split('\t')
                assert tag in ['1', '-1']
                labels.append(int(tag == '1'))
                inputs.append(body)
        
        assert len(inputs) == len(labels)
        
        return inputs, labels

