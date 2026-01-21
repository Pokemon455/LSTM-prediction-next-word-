import pandas as pd
import tiktoken
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import re

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 1. Load dataset (assume Kaggle path - change if needed)
df = pd.read_csv("/kaggle/input/chabot-qa-dataset/General_question_answer_dataset.csv")  # Change to your actual CSV path
# Assuming columns: 'question' and 'answer'
answers = df["answer"].tolist()  # Use answers for training, or combine question+answer
for answer in answers:
    text = re.sub(r'([.,!?:;])\1+', r'\1', answer)

# 2. Tokenization using tiktoken (cl100k_base like GPT-4)
enc = tiktoken.get_encoding("cl100k_base")
vocab_size = enc.n_vocab
print(f"Vocab size: {vocab_size}")

def tokenize_function(text_list):
    all_tokens = []
    for text in text_list:
        tokens = enc.encode(text)
        all_tokens.extend(tokens)
    return torch.tensor(all_tokens, dtype=torch.long)

all_tokens = tokenize_function(text)  # Or combine question + answer
print(f"Total tokens: {len(all_tokens)}")

# 3. Create sequences (chunks) for language modeling (next token prediction)
seq_length = 512  # Adjust as per your memory

class TokenDataset(Dataset):
    def __init__(self, tokens, seq_length):
        self.seq_length = seq_length
        self.sequences = []
        
        # Create overlapping chunks
        for i in range(0, len(tokens) - seq_length):
            x = tokens[i : i + seq_length]
            y = tokens[i + 1 : i + seq_length + 1]
            self.sequences.append((x, y))
        
        print(f"Total sequences created: {len(self.sequences)}")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        x, y = self.sequences[idx]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

dataset = TokenDataset(all_tokens, seq_length)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Check dataloader
for batch_x, batch_y in dataloader:
    print(f"Batch X shape: {batch_x.shape}")  # [32, 512]
    print(f"Batch Y shape: {batch_y.shape}")  # [32, 512]
    break

# 4. Simple LSTM Model
class SimpleLSTMGenerator(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, hidden_size=512):
        super().__init__()
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # LSTM - basic
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )
        
        # Linear layers chain
        self.fc1 = nn.Linear(hidden_size, hidden_size * 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_size * 2, hidden_size)
        self.fc3 = nn.Linear(hidden_size, vocab_size)  # Output logits for vocab

    def forward(self, x):
        # x: [batch, seq_len] -> token ids
        
        embedded = self.embedding(x)  # [batch, seq_len, embed_dim]
        
        lstm_out, _ = self.lstm(embedded)  # [batch, seq_len, hidden_size]
        
        out = lstm_out.reshape(-1, lstm_out.size(-1))  # Flatten to [batch*seq_len, hidden]
        
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        logits = self.fc3(out)  # [batch*seq_len, vocab_size]
        
        # Reshape back for loss
        batch_size, seq_len, _ = lstm_out.shape
        logits = logits.view(batch_size, seq_len, -1)
        
        return logits

# Initialize model
model = SimpleLSTMGenerator(vocab_size=vocab_size).to(device)

# 5. Optimizer & Loss
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding if any, else remove ignore_index

# 6. Training Loop
epochs = 80
for epoch in range(epochs):
    model.train()
    total_loss = 0.0
    
    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        
        optimizer.zero_grad()
        
        logits = model(batch_x)  # [batch, seq_len, vocab_size]
        
        # Flatten for loss
        loss = criterion(logits.view(-1, vocab_size), batch_y.view(-1))
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    print(f"Epoch [{epoch+1}/{epochs}] - Avg Loss: {avg_loss:.4f}")

# 7. Basic Generation Function (demo)
def generate_text(model, start_text, max_length=100, temperature=0.8):
    model.eval()
    with torch.no_grad():
        tokens = enc.encode(start_text)
        input_tensor = torch.tensor([tokens]).to(device)
        generated = tokens.copy()
        
        for _ in range(max_length):
            logits = model(input_tensor)
            logits = logits[:, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
            generated.append(next_token)
            input_tensor = torch.tensor([generated]).to(device)
        
        return enc.decode(generated)

# Example usage
sample = generate_text(model, "Hello, how are you", max_length=50)
print("Generated text:", sample)

# Save model (optional)
torch.save(model.state_dict(), "lstm_language_model.pth")