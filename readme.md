<div align="center">

# 🧠 LSTM Next Word Prediction

**A custom LSTM neural network that predicts the next word in any sentence — built from scratch with PyTorch & GPT-4 tokenizer.**

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![tiktoken](https://img.shields.io/badge/Tokenizer-tiktoken%20cl100k-412991?style=for-the-badge)](https://github.com/openai/tiktoken)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

> *Like your phone keyboard suggestions — but powered by a real LSTM trained from scratch.*

</div>

---

## ✨ What It Does

Type any partial sentence → the model predicts the **most likely next words**.

```
Input:  "The weather today is"
Output: "The weather today is sunny and warm" ✅
```

---

## 🏗️ Architecture

```
Raw Text
   │
   ▼
tiktoken (cl100k_base)        ← Same tokenizer as GPT-4
   │
   ▼
Token Embeddings (256-dim)
   │
   ▼
LSTM Layer (512 hidden units)
   │
   ▼
FC Layers  →  ReLU  →  Dropout(0.3)
   │
   ▼
Softmax → Next Token
```

| Component | Detail |
|-----------|--------|
| **Tokenizer** | tiktoken `cl100k_base` (GPT-4 style) |
| **Embedding** | 256-dim learned embeddings |
| **LSTM** | 512 hidden units, 1 layer |
| **Classifier** | 3-layer FC with ReLU + Dropout |
| **Dataset** | Chatbot Q&A pairs (CSV) |
| **Hardware** | Auto GPU/CPU detection |

---

## 🛠️ Tech Stack

| Library | Purpose |
|---------|---------|
| `PyTorch` | LSTM model & training loop |
| `tiktoken` | GPT-4 style tokenization |
| `Pandas` | Dataset loading & preprocessing |
| `NumPy` | Numerical operations |

---

## 🚀 Quick Start

### 1. Clone & Install
```bash
git clone https://github.com/Pokemon455/LSTM-prediction-next-word-.git
cd LSTM-prediction-next-word-
pip install -r requirements.txt
```

### 2. Train the Model
```bash
python train.py
```
> 💡 **Kaggle users:** Update the dataset path in `train.py` to `/kaggle/input/your-dataset/`

This will save `best_model.pth` in the project folder.

### 3. Run Inference
```bash
python inference.py
```

```
🧠 LSTM Text Generator
   Device  : cuda
   Vocab   : 100,277 tokens
   Type 'exit' to quit

Prompt: The weather today is
Output: The weather today is sunny and pleasant outside
```

---

## 📂 Project Structure

```
LSTM-prediction-next-word-/
├── train.py          # Training script (LSTM model + training loop)
├── inference.py      # Standalone inference & interactive CLI
├── Dataset.csv       # Q&A training data
├── requirements.txt  # Dependencies
└── README.md         # Documentation
```

---

## 💡 Key Features

- ⚡ **GPU ready** — auto-detects CUDA, falls back to CPU
- 🔤 **GPT-4 tokenizer** — tiktoken cl100k_base (100k+ vocab)
- 🧹 **Clean preprocessing** — removes duplicate punctuation
- 🎛️ **Temperature sampling** — control creativity of output
- 🛡️ **Error handling** — clear messages for missing model weights
- 📦 **Kaggle compatible** — easy path config for cloud training

---

## 🗺️ Roadmap

- [x] LSTM training from scratch
- [x] GPT-4 tokenizer integration
- [x] Standalone inference CLI
- [ ] Pre-trained model weights upload
- [ ] Streamlit / Gradio demo UI
- [ ] Beam search decoding
- [ ] Multi-layer LSTM support

---

## 👤 Author

**Arbaz** — AI/ML Developer
🔗 [GitHub](https://github.com/Pokemon455)

---

<div align="center">

⭐ **If this helped you, drop a star — it keeps the project alive!** ⭐

</div>
