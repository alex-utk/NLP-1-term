import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

class BertClassifier(nn.Module):
    """Модель классификатора c bert-small энкодером"""

    def __init__(self: nn.Module) -> None:
        super(BertClassifier, self).__init__()

        self.encoder = AutoModel.from_pretrained("prajjwal1/bert-small")

        self.fc1 = nn.Linear(512, 300)
        self.bn1 = nn.BatchNorm1d(300)
        self.dropout1 = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(300, 100)
        self.bn2 = nn.BatchNorm1d(100)
        self.dropout2 = nn.Dropout(p=0.2)
        self.fc3 = nn.Linear(100, 1)

        self.init_weights()


    def init_weights(self: nn.Module) -> None:
        for param in self.encoder.parameters():
            param.requires_grad = False

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)

        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
        nn.init.zeros_(self.fc3.bias)


    def forward(self: nn.Module,
                input_ids: torch.Tensor,
                token_type_ids: torch.Tensor,
                attention_mask: torch.Tensor
                ) -> torch.Tensor:
        x = self.encoder(input_ids=input_ids,
                         token_type_ids=token_type_ids,
                         attention_mask=attention_mask).last_hidden_state[:,0,:]
        x = F.normalize(x)

        x = F.elu(self.fc1(x))
        x = self.bn1(x)
        x = self.dropout1(x)

        x = F.elu(self.fc2(x))
        x = self.bn2(x)
        x = self.dropout2(x)

        x = F.sigmoid(self.fc3(x))  # что лучше? использовать nn.Sigmoid или F.sigmoid и т.п.

        return x
