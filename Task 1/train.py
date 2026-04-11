import torch
from dataset_tokenizer import get_dataloaders
from model import GPTLanguageModel


batch_size = 16
block_size = 64    
max_iters = 8500    
eval_interval = 500  
learning_rate = 3e-4
eval_iters = 100
n_embd = 128         
n_head = 4           
n_layer = 4
dropout = 0.3
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"Using device: {device}")


print("Loading data...")
train_loader, val_loader, dataset = get_dataloaders(block_size, batch_size)
vocab_size = dataset.vocab_size


model = GPTLanguageModel(vocab_size, n_embd, n_head, n_layer, block_size, dropout)
model = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-2)


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split, loader in [('train', train_loader), ('val', val_loader)]:
        losses = torch.zeros(eval_iters)
        loader_iter = iter(loader)
        for k in range(eval_iters):
            try:
                X, Y = next(loader_iter)
            except StopIteration:
                loader_iter = iter(loader)
                X, Y = next(loader_iter)
            X, Y = X.to(device), Y.to(device)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


print("Starting training...")
train_iter = iter(train_loader)
for iter_num in range(max_iters):

    if iter_num % eval_interval == 0 or iter_num == max_iters - 1:
        losses = estimate_loss()
        print(f"Step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")


    try:
        xb, yb = next(train_iter)
    except StopIteration:
        train_iter = iter(train_loader)
        xb, yb = next(train_iter)
    
    xb, yb = xb.to(device), yb.to(device)


    logits, loss = model(xb, yb)
    
    
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()





torch.save(model.state_dict(), 'shakespeare_bpe_v1.pt')
print("Model saved successfully!")



model.eval()

context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated_indices = model.generate(context, max_new_tokens=500)[0].tolist()
print(dataset.decode(generated_indices))
