import time
import nltk
import torch
import numpy as np
import torch.nn as nn
from cleantext import clean
from gensim.utils import tokenize
from torch.utils.data import DataLoader
from torch.nn.functional import softmax
from torchmetrics.classification import BinaryRecall, BinaryPrecision, BinaryF1Score


nltk.download('wordnet')
wnl = nltk.WordNetLemmatizer()


def clean_func(text: str) -> str:
    """Препроцессинг текста

    Args:
        text (str): текст

    Returns:
        str: обработанный текст
    """
    text = clean(text,
                 no_line_breaks=True,
                 no_urls=True,
                 no_emails=True,
                 no_phone_numbers=True,
                 no_numbers=True,
                 no_digits=True,
                 no_currency_symbols=True,
                 no_punct=True,
                 no_emoji=True,
                 replace_with_punct=" ",
                 replace_with_url=" ",
                 replace_with_email=" ",
                 replace_with_phone_number=" ",
                 replace_with_number=" ",
                 replace_with_digit=" ",
                 replace_with_currency_symbol=" ",
                 lang="en")
    text = ' '.join([wnl.lemmatize(word) for word in tokenize(text)])
    return text


def train_model(
    model: nn.Module,
    train_loader: nn.CrossEntropyLoss,
    valid_loader: DataLoader,
    criterion: torch.nn.CrossEntropyLoss,
    sheduler: torch.optim.lr_scheduler,
    optimizer: torch.optim.Optimizer,
    n_epochs: int,
    filename: str,
    patience: int = 5
) -> None:
    """Запуск обучения модели

    Args:
        model (nn.Module): модель
        train_loader (nn.CrossEntropyLoss): loader с тестовыми данными
        valid_loader (DataLoader): loader с валидационными данными
        criterion (torch.nn.CrossEntropyLoss): loss
        sheduler (torch.optim.lr_scheduler): sheduler
        optimizer (torch.optim.Optimizer): optimizer
        n_epochs (int): количество эпох
        filename (str): куда сохранять веса модели
        patience (int, optional): сколько эпох лосс можкет расти перед остановокой обучения. Defaults to 5.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.train()
    model.to(device)
    valid_loss_min = np.Inf
    current_p = 0

    for epoch in range(1, n_epochs + 1):
        print(time.ctime(), 'Epoch:', epoch)

        # train
        train_loss = []
        for data, target in train_loader:
            optimizer.zero_grad()
            target = target.to(device)
            
            output = model(**{k: v.to(device) for k, v in data.items()})
            
            loss = criterion(output, target)
            train_loss.append(loss.item())
            loss.backward()
            optimizer.step()

        # validation
        model.eval()
        val_loss = []
        for data, target in valid_loader:
            target = target.to(device)

            with torch.inference_mode():
                output = model(**{k: v.to(device) for k, v in data.items()})
            loss = criterion(output, target)
            val_loss.append(loss.item())

        valid_loss = np.mean(val_loss)
        print(f'Epoch {epoch}, train loss: {np.mean(train_loss):.4f}, valid loss: {valid_loss:.4f}.')

        sheduler.step(valid_loss)
        # если лосс стал меньше, то сохраняем чекпоинт, а терпение сбрасываем
        if valid_loss <= valid_loss_min:
            print(f'Validation loss decreased ({valid_loss_min:.6f} --> {valid_loss:.6f}). Saving model.')
            torch.save(model.state_dict(), filename)
            valid_loss_min = valid_loss
            current_p = 0
        # если лосс стал больше, то терпение нарастает
        else:
            current_p += 1
            print(f'{current_p} epochs of increasing val loss')
            if current_p > patience:
                print('Stopping training')
                break

  
def test_model(
    model: nn.Module,
    test_loader: DataLoader,
) -> tuple[float, float]:
    """_summary_

    Args:
        model (nn.Module): модель
        test_loader (DataLoader): загрузчик тестовых данных
        n_classes (int): количество классов

    Returns:
        tuple(float, float): значения precision, recall
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()

    recall = BinaryRecall()
    precision = BinaryPrecision()
    f1 = BinaryF1Score()

    targets = []
    preds = []

    for data, label in test_loader:
        label = label.cpu()
        with torch.inference_mode():
            output = model(**{k: v.to(device) for k, v in data.items()})
            output = output.cpu()

        targets.append(label.item())
        preds.append(output.item())

    targets = torch.tensor(targets, dtype=torch.float32)
    preds = torch.tensor(preds, dtype=torch.float32)

    precision_score = precision(preds, targets).item()
    recall_score = recall(preds, targets).item()
    f1_score = f1(preds, targets).item()

    return precision_score, recall_score, f1_score
