import torch
from torch import nn
from torch.utils.data import DataLoader
from torch import optim
from tqdm import tqdm

# padding token index
pad_idx_src = src_vocab["<pad>"]
pad_idx_tgt = tgt_vocab["<pad>"]

# Transformer model
model = Transformer(
    src_vocab_size=len(src_vocab),
    tgt_vocab_size=len(tgt_vocab),
    d_model=512,
    num_heads=8,
    num_layers=6,
    dropout=0.1
).to(device)

# loss function and optimizer
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx_tgt)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Generating mask
def make_src_mask(src, pad_idx):
    return (src != pad_idx).unsqueeze(1).unsqueeze(2)  # [B, 1, 1, src_len]

def make_tgt_mask(tgt, pad_idx):
    B, T = tgt.size()
    pad_mask = (tgt != pad_idx).unsqueeze(1).unsqueeze(2)  # [B, 1, 1, T]
    look_ahead_mask = torch.tril(torch.ones((T, T), device=tgt.device)).bool()
    look_ahead_mask = look_ahead_mask.unsqueeze(0).unsqueeze(1)  # [1, 1, T, T]
    return pad_mask & look_ahead_mask

# Validation data preprocessing and generating DataLoader
val_src_sentences, val_tgt_sentences = preprocess(val_data)
val_dataset = TranslationDataset(val_src_sentences, val_tgt_sentences, src_vocab, tgt_vocab)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

# Result lists
train_losses = []
val_losses = []
val_accuracies = []

# The number of epochs
NUM_EPOCHS = 100

# Training
for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0

    for src_batch, tgt_batch in tqdm(train_loader, desc=f"[Train Epoch {epoch+1}]"):
        src = src_batch.to(device)
        tgt_input = tgt_batch[:, :-1].to(device)
        tgt_output = tgt_batch[:, 1:].to(device)

        src_mask = make_src_mask(src, pad_idx_src)
        tgt_mask = make_tgt_mask(tgt_input, pad_idx_tgt)

        # forward propagation
        output = model(src, tgt_input, src_mask=src_mask, tgt_mask=tgt_mask)
        output = output.reshape(-1, output.shape[-1])
        tgt_output = tgt_output.reshape(-1)

        # Loss calculation and backward propagation
        loss = criterion(output, tgt_output)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    print(f"[Epoch {epoch+1}] Train Loss: {avg_train_loss:.4f}")

    # Validation evaluation 
    model.eval()
    val_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for src_batch, tgt_batch in val_loader:
            src = src_batch.to(device)
            tgt_input = tgt_batch[:, :-1].to(device)
            tgt_output = tgt_batch[:, 1:].to(device)

            src_mask = make_src_mask(src, pad_idx_src)
            tgt_mask = make_tgt_mask(tgt_input, pad_idx_tgt)

            output = model(src, tgt_input, src_mask=src_mask, tgt_mask=tgt_mask)  # [B, T, vocab]

            # Validation loss
            output_flat = output.reshape(-1, output.shape[-1])
            tgt_flat   = tgt_output.reshape(-1)
            loss = criterion(output_flat, tgt_flat)
            val_loss += loss.item()

            # Validation accuracy
            preds   = output.argmax(dim=-1)           # [B, T]
            non_pad = tgt_output != pad_idx_tgt
            correct += ((preds == tgt_output) & non_pad).sum().item()
            total   += non_pad.sum().item()

    avg_val_loss = val_loss / len(val_loader)
    accuracy     = correct / total

    val_losses.append(avg_val_loss)
    val_accuracies.append(accuracy)
    