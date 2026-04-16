import torch
import torch.nn as nn
import tiktoken
import sys
import os

# ─── Device Setup ────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── Tokenizer ───────────────────────────────────────────────────
enc = tiktoken.get_encoding("cl100k_base")
VOCAB_SIZE = enc.n_vocab
SEQ_LENGTH = 512

# ─── Model Definition ────────────────────────────────────────────
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


# ─── Load Model ──────────────────────────────────────────────────
def load_model(model_path: str = "best_model.pth") -> LSTMTextGenerator:
    """Load trained LSTM model from disk."""
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        print("   Please run train.py first to generate the model weights.")
        sys.exit(1)

    model = LSTMTextGenerator(vocab_size=VOCAB_SIZE)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"✅ Model loaded from {model_path} | Device: {device}")
    return model


# ─── Text Generation ─────────────────────────────────────────────
def generate(
    model: LSTMTextGenerator,
    prompt: str,
    max_tokens: int = 50,
    temperature: float = 0.8
) -> str:
    """
    Generate text given a prompt.

    Args:
        model: Trained LSTMTextGenerator instance
        prompt: Starting text for generation
        max_tokens: Number of tokens to generate
        temperature: Sampling temperature (0.5 = conservative, 1.0 = creative)

    Returns:
        Generated text string
    """
    if not prompt.strip():
        raise ValueError("Prompt cannot be empty.")

    with torch.no_grad():
        tokens = enc.encode(prompt)
        for _ in range(max_tokens):
            inp = torch.tensor([tokens[-SEQ_LENGTH:]]).to(device)
            logits = model(inp)[:, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, 1).item()
            tokens.append(next_tok)

    return enc.decode(tokens)


# ─── Interactive Mode ────────────────────────────────────────────
if __name__ == "__main__":
    print("\n🧠 LSTM Text Generator")
    print(f"   Device  : {device}")
    print(f"   Vocab   : {VOCAB_SIZE:,} tokens")
    print("   Type \'exit\' to quit\n")

    model = load_model("best_model.pth")

    while True:
        try:
            prompt = input("Prompt: ").strip()
            if prompt.lower() == "exit":
                print("Goodbye!")
                break
            if not prompt:
                print("⚠️  Please enter a prompt.\n")
                continue

            output = generate(model, prompt, max_tokens=40, temperature=0.8)
            print(f"Output: {output}\n")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}\n")
