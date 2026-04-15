import torch
import tiktoken
from train import LSTMTextGenerator

# ─── Load Model ──────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
enc = tiktoken.get_encoding("cl100k_base")

model = LSTMTextGenerator(vocab_size=enc.n_vocab)
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.to(device)
model.eval()

print("✅ Model loaded!")

# ─── Generate Text ───────────────────────────────────────────────
def generate(prompt, max_tokens=50, temperature=0.8):
    """
    Generate next words given a prompt.
    
    Args:
        prompt (str): Starting text
        max_tokens (int): How many tokens to generate
        temperature (float): 0.5 = conservative, 1.0 = creative
    
    Returns:
        str: Generated text
    """
    with torch.no_grad():
        tokens = enc.encode(prompt)
        for _ in range(max_tokens):
            inp = torch.tensor([tokens[-512:]]).to(device)
            logits = model(inp)[:, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, 1).item()
            tokens.append(next_tok)
    return enc.decode(tokens)

# ─── Interactive Mode ────────────────────────────────────────────
if __name__ == "__main__":
    print("\n🧠 LSTM Text Generator — Type your prompt!")
    print("Type \'exit\' to quit\n")
    
    while True:
        prompt = input("Prompt: ").strip()
        if prompt.lower() == "exit":
            break
        result = generate(prompt, max_tokens=40, temperature=0.8)
        print(f"Output: {result}\n")
