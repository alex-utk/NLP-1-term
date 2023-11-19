import torch
import pandas as pd
from torch.utils.data import Dataset
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-small")


def collate_fn(batch):
    texts = [item[0] for item in batch]
    labels = torch.FloatTensor([item[1] for item in batch])
    labels = torch.unsqueeze(labels, 1)

    features_dict = tokenizer(texts,
                              padding=True,
                              return_tensors='pt')
    return features_dict, labels


class TextDataset(Dataset):
    """Класс датасета для работы с one-how-encoded векторами"""

    def __init__(self: Dataset, df: pd.DataFrame) -> None:
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data = list(df.itertuples(index=False))

    def __len__(self: Dataset) -> int:
        return len(self.data)

    def __getitem__(self: Dataset, idx: int) -> tuple[torch.Tensor, int]:
        row = self.data[idx]
        label = int(row.label)

        return row.text, label
