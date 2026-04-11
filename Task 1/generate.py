import torch
from dataset_tokenizer import load_dataset, ShakespeareDataset
from model import GPTLanguageModel


block_size = 64
n_embd = 128
n_head = 4
n_layer = 4
dropout = 0.2
device = 'cuda' if torch.cuda.is_available() else 'cpu'

text = load_dataset(print_text=False)

dataset = ShakespeareDataset(text, block_size)
vocab_size = dataset.vocab_size

model = GPTLanguageModel(vocab_size, n_embd, n_head, n_layer, block_size, dropout)


model.load_state_dict(torch.load('shakespeare_bpe_v1.pt', map_location=device, weights_only=True))
model.to(device)
model.eval()


context = torch.tensor([[50256]], dtype=torch.long, device=device)

current_temperature = float(input("Enter temperature: "))

print(f"Generating 1000 characters: ")

generated_indices = model.generate(context, max_new_tokens=1000, temperature=current_temperature)[0].tolist()
print(dataset.decode(generated_indices))