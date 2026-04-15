import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn


PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"


@dataclass
class TrainingConfig:
    context_size: int = 3
    embedding_dim: int = 64
    hidden_dim: int = 128


class NextWordMLP(nn.Module):
    def __init__(self, vocab_size: int, context_size: int, embedding_dim: int, hidden_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.mlp = nn.Sequential(
            nn.Linear(context_size * embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, vocab_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(x)
        features = emb.view(emb.size(0), -1)
        return self.mlp(features)


def simple_tokenize(text: str) -> List[str]:
    return [tok.strip().lower() for tok in text.replace("\n", " ").split() if tok.strip()]


def build_vocab(texts: List[str], min_freq: int = 1) -> Tuple[Dict[str, int], Dict[int, str]]:
    counter = Counter()
    for text in texts:
        counter.update(simple_tokenize(text))

    vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for token, freq in counter.items():
        if freq >= min_freq:
            vocab[token] = len(vocab)

    inv_vocab = {idx: token for token, idx in vocab.items()}
    return vocab, inv_vocab


def encode_tokens(tokens: List[str], vocab: Dict[str, int]) -> List[int]:
    unk_idx = vocab[UNK_TOKEN]
    return [vocab.get(token, unk_idx) for token in tokens]


def create_training_examples(texts: List[str], vocab: Dict[str, int], context_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    x_data: List[List[int]] = []
    y_data: List[int] = []

    for text in texts:
        ids = encode_tokens(simple_tokenize(text), vocab)
        if len(ids) <= context_size:
            continue

        for i in range(context_size, len(ids)):
            x_data.append(ids[i - context_size : i])
            y_data.append(ids[i])

    if not x_data:
        raise ValueError("No training examples were created. Reduce context_size or provide more data.")

    return torch.tensor(x_data, dtype=torch.long), torch.tensor(y_data, dtype=torch.long)


def save_artifacts(model: NextWordMLP, vocab: Dict[str, int], config: TrainingConfig, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "model.pt")

    metadata = {
        "vocab": vocab,
        "context_size": config.context_size,
        "embedding_dim": config.embedding_dim,
        "hidden_dim": config.hidden_dim,
    }
    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def load_artifacts(out_dir: Path, device: torch.device) -> Tuple[NextWordMLP, Dict[str, int], Dict[int, str], TrainingConfig]:
    with open(out_dir / "metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)

    vocab = metadata["vocab"]
    inv_vocab = {idx: tok for tok, idx in vocab.items()}
    config = TrainingConfig(
        context_size=metadata["context_size"],
        embedding_dim=metadata["embedding_dim"],
        hidden_dim=metadata["hidden_dim"],
    )

    model = NextWordMLP(
        vocab_size=len(vocab),
        context_size=config.context_size,
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
    )
    state = torch.load(out_dir / "model.pt", map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, vocab, inv_vocab, config


def generate_text(
    model: NextWordMLP,
    seed_text: str,
    vocab: Dict[str, int],
    inv_vocab: Dict[int, str],
    context_size: int,
    max_new_tokens: int = 20,
    temperature: float = 1.0,
    device: torch.device | None = None,
) -> str:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokens = simple_tokenize(seed_text)
    if not tokens:
        tokens = [UNK_TOKEN] * context_size

    while len(tokens) < context_size:
        tokens.insert(0, PAD_TOKEN)

    generated = tokens.copy()

    for _ in range(max_new_tokens):
        context_tokens = generated[-context_size:]
        x = torch.tensor([encode_tokens(context_tokens, vocab)], dtype=torch.long, device=device)
        with torch.no_grad():
            logits = model(x).squeeze(0) / max(temperature, 1e-6)
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, 1).item()

        next_token = inv_vocab.get(next_id, UNK_TOKEN)
        generated.append(next_token)

    return " ".join(generated)
