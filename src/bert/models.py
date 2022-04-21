import torch
from torch import nn

from transformers import BertModel

BertLayerNorm = torch.nn.LayerNorm
class PosNegBERT(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        
        self.bert = BertModel.from_pretrained(model_name)
        self.fc1 = nn.Linear(768, 500)
        self.fc2 = nn.Linear(500, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, input_ids, attention_mask, token_type_ids):
        output_dict = {}
        num_batch = input_ids.shape[0]

        input_ids = input_ids.view(num_batch, -1) # (batch_size, 1, seq_len) -> (batch_size, seq_len)
        token_type_ids = token_type_ids.view(num_batch, -1)
        attention_mask = attention_mask.view(num_batch, -1)
        _, pooled = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask,
                               output_hidden_states=False, output_attentions=False) # (batch_size, hidden_size)
        x = self.relu(self.fc1(pooled)) # -> (batch_size, 500)
        output = self.sigmoid(self.fc2(x)) # -> (batch_size, 1)
        output = output.view(num_batch, -1)

        output_dict["logits"] = output

        return output_dict
    

