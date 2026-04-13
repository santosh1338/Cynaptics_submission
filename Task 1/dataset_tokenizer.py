
import requests
import os
import torch
from torch.utils.data import Dataset, DataLoader
import tiktoken

url = "https://raw.githubusercontent.com/karpathy/char-rnn/refs/heads/master/data/tinyshakespeare/input.txt"
DATA_PATH = "shakespeare.txt"

def download_dataset() -> None:
    if os.path.exists(DATA_PATH):
        return


    text_file = requests.get(url).text
    with open(DATA_PATH, "w", encoding="utf-8") as f:
        f.write(text_file)

def load_dataset(print_text=False) -> str:
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        txt = f.read()

    if print_text:
        print('Dataset loaded successfully.')

    return txt

class ShakespeareDataset(Dataset):
    def __init__(self, text, block_size):
        self.text = text
        self.block_size = block_size
        
        
        self.tokenizer = tiktoken.get_encoding("gpt2")
        self.vocab_size = self.tokenizer.n_vocab 
        
        self.data = torch.tensor(self.encode(text), dtype=torch.long)

    def encode(self, s):
        
        return self.tokenizer.encode(s, allowed_special={"<|endoftext|>"})

    def decode(self, l):
        
        return self.tokenizer.decode(l)

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        chunk = self.data[idx : idx + self.block_size + 1]
        x = chunk[:-1]
        y = chunk[1:]
        return x, y

def get_dataloaders(block_size, batch_size, train_split=0.9):
    download_dataset()
    text = load_dataset(print_text=False)

    
    n = int(train_split * len(text))
    train_text = text[:n]
    val_text = text[n:]

    
    train_dataset = ShakespeareDataset(train_text, block_size)
    val_dataset = ShakespeareDataset(val_text, block_size)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, train_dataset

if __name__ == "__main__":

    print("Testing...")
    train_dl, val_dl, ds = get_dataloaders(block_size=8, batch_size=4)
