import pandas as pd
import tiktoken
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import re

# ─── Device Setup ───────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ─── Load Dataset ───────────────────────────────────────────────
# Change path if running locally (default: Kaggle)
df = pd.read_csv("Dataset.csv")
answers = df["answer"].tolist()

texts = []
for answer in answers:
    text = re.sub(r'([.,!?:;])\1+', r'\1', answer)
    texts.append(text)

# ─── Tokenization (GPT-4 tokenizer) ─────────────────────────────
enc = tiktoken.get_encoding("cl100k_base")
vocab_size = enc.n_vocab
print(f"Vocab size: {vocab_size}")

def tokenize(text_list):
    all_tokens = []
    for text in text_list:
        all_tokens.extend(enc.encode(text))
    return torch.tensor(all_tokens, dtype=torch.long)

all_tokens = tokenize(texts)
print(f"Total tokens: {len(all_tokens)}")

# ─── Dataset Class ───────────────────────────────────────────────
SEQ_LENGTH = 512

class TokenDataset(Dataset):
    def __init__(self, tokens, seq_length):
        self.sequences = []
        for i in range(0, len(tokens) - seq_length):
            x = tokens[i: i + seq_length]
            y = tokens[i + 1: i + seq_length + 1]
            self.sequences.append((x, y))
        print(f"Total sequences: {len(self.sequences)}")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        x, y = self.sequences[idx]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

dataset = TokenDataset(all_tokens, SEQ_LENGTH)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# ─── LSTM Model ─────────────────────────────────────────────────
class LSTMTextGenerator(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, hidden_size=512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True
        )
        self.fc1 = nn.Linear(hidden_size, hidden_size * 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_size * 2, hidden_size)
        self.fc3 = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        out = lstm_out.reshape(-1, lstm_out.size(-1))
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        logits = self.fc3(out)
        b, s, _ = lstm_out.shape
        return logits.view(b, s, -1)

model = LSTMTextGenerator(vocab_size).to(device)

# ─── Training ────────────────────────────────────────────────────
optimizer = optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.CrossEntropyLoss()

EPOCHS = 15
best_loss = float('inf')

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0

    for batch_x, batch_y in dataloader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()
        logits = model(batch_x)
        loss = criterion(logits.view(-1, vocab_size), batch_y.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {avg_loss:.4f}")

    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), "best_model.pth")
        print(f"  ✅ Saved best model (loss: {best_loss:.4f})")

# ─── Text Generation ─────────────────────────────────────────────
def generate(prompt, max_tokens=50, temperature=0.8):
    model.eval()
    with torch.no_grad():
        tokens = enc.encode(prompt)
        for _ in range(max_tokens):
            inp = torch.tensor([tokens[-512:]]).to(device)
            logits = model(inp)[:, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, 1).item()
            tokens.append(next_tok)
    return enc.decode(tokens)

print("\n--- Sample Output ---")
print(generate("Hello, how are you", max_tokens=30))

torch.save(model.state_dict(), "lstm_model.pth")
print("\nModel saved as lstm_model.pth")
