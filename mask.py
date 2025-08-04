import torch

# Device Setting
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Generating mask
def make_src_mask(src, pad_idx):
    return (src != pad_idx).unsqueeze(1).unsqueeze(2)  # [B, 1, 1, src_len]

def make_tgt_mask(tgt, pad_idx):
    B, T = tgt.size()

    # Padding mask
    pad_mask = (tgt != pad_idx).unsqueeze(1).unsqueeze(2)  # [B, 1, 1, T]

    # Look-ahead mask
    look_ahead_mask = torch.tril(torch.ones((T, T), device=tgt.device)).bool()  # [T, T]
    look_ahead_mask = look_ahead_mask.unsqueeze(0).unsqueeze(1)  # [1, 1, T, T]

    return pad_mask & look_ahead_mask # [B, 1, T, T]

# Transformer model
model = Transformer(
    src_vocab_size=len(src_vocab),
    tgt_vocab_size=len(tgt_vocab),
    d_model=512,
    num_heads=8,
    num_layers=6,
    dropout=0.1
).to(device)

# Example batch to check Forward pass
pad_idx_src = src_vocab["<pad>"]
pad_idx_tgt = tgt_vocab["<pad>"]

# A batch for the model
for src_batch, tgt_batch in train_loader:
    # input: src, tgt_input / output: tgt_output
    src = src_batch.to(device)
    tgt_input = tgt_batch[:, :-1].to(device)
    tgt_output = tgt_batch[:, 1:].to(device)

    src_mask = make_src_mask(src, pad_idx_src).to(device)
    tgt_mask = make_tgt_mask(tgt_input, pad_idx_tgt).to(device)

    # Output verification
    with torch.no_grad():
      output = model(src, tgt_input, src_mask=src_mask, tgt_mask=tgt_mask)
    print("Output shape:", output.shape) # [B, tgt_len-1, vocab]
    print("Target shape:", tgt_output.shape) # [B, tgt_len-1]

    break
