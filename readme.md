# 🧠 LSTM Next Word Prediction

> Predict the next word in any sentence using a custom-built LSTM model trained on a Q&A dataset — built with PyTorch & tiktoken (GPT-4 tokenizer).

---

## 🎯 What It Does

Type any partial sentence → model predicts the **most likely next word**.  
Works like your phone keyboard suggestions, but powered by a real LSTM neural network trained from scratch.

```
Input:  "The weather today is"
Output: "sunny" ✅
```

---

## 🏗️ Architecture

```
Text → tiktoken (cl100k_base) → Token IDs → LSTM Layers → Linear → Softmax → Next Word
```

- **Tokenizer:** tiktoken `cl100k_base` (same as GPT-4)
- **Model:** Multi-layer LSTM with PyTorch
- **Dataset:** Chatbot Q&A dataset (General question-answer pairs)
- **Training:** GPU/CPU auto-detect via `torch.device`

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| `PyTorch` | LSTM model & training loop |
| `tiktoken` | GPT-4 style tokenization |
| `Pandas` | Dataset loading & preprocessing |
| `NumPy` | Numerical operations |

---

## 🚀 Quick Start

### 1. Clone the repo
```bash
git clone https://github.com/Pokemon455/LSTM-prediction-next-word-.git
cd LSTM-prediction-next-word-
```

### 2. Install requirements
```bash
pip install -r requirements.txt
```

### 3. Add your dataset
Place your CSV file in the project folder with `question` and `answer` columns.  
Or use the included `Dataset.csv`.

### 4. Train the model
```bash
python train.py
```

> 💡 **Kaggle users:** Update the dataset path in `train.py` to `/kaggle/input/your-dataset/`

---

## 📂 Project Structure

```
LSTM-prediction-next-word-/
├── train.py          # Main training script
├── Dataset.csv       # Q&A training data
├── requirements.txt  # Dependencies
└── README.md         # Documentation
```

---

## 💡 Key Features

- ✅ GPU support (auto-detects CUDA)
- ✅ GPT-4 tokenizer (tiktoken cl100k_base)
- ✅ Clean text preprocessing (removes duplicate punctuation)
- ✅ Kaggle-ready training script

---

## 🌱 Future Improvements

- [ ] Add inference script for real-time prediction
- [ ] Streamlit/Gradio demo UI
- [ ] Beam search for better predictions
- [ ] Pre-trained model weights upload

---

## 👤 Author

**Pokemon455** — AI/ML Developer  
🔗 [GitHub Profile](https://github.com/Pokemon455)

---

⭐ **If this helped you, please give it a star!**
