import argparse
from pathlib import Path

import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset, random_split

from mlp_model import (
    NextWordMLP,
    TrainingConfig,
    build_vocab,
    create_training_examples,
    save_artifacts,
)


def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    df = pd.read_csv(args.data_path)
    texts = (df["question"].fillna("") + " " + df["answer"].fillna("")).tolist()

    config = TrainingConfig(
        context_size=args.context_size,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
    )

    vocab, _ = build_vocab(texts, min_freq=args.min_freq)
    x, y = create_training_examples(texts, vocab, context_size=config.context_size)
    dataset = TensorDataset(x, y)

    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    model = NextWordMLP(
        vocab_size=len(vocab),
        context_size=config.context_size,
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_val_loss = float("inf")
    output_dir = Path(args.output_dir)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_loss = running_loss / max(1, len(train_loader))

        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                val_running_loss += loss.item()

        val_loss = val_running_loss / max(1, len(val_loader))
        print(f"Epoch {epoch}/{args.epochs} - train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_artifacts(model, vocab, config, output_dir)
            print(f"Saved best model to {output_dir} (val_loss={best_val_loss:.4f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a simple MLP for next-word prediction")
    parser.add_argument("--data-path", default="Dataset.csv")
    parser.add_argument("--output-dir", default="artifacts")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--context-size", type=int, default=3)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--min-freq", type=int, default=1)
    parser.add_argument("--val-split", type=float, default=0.1)

    args = parser.parse_args()
    train(args)
