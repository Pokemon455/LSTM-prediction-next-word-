# 🧠 LSTM Next Word Prediction

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-Latest-red?style=for-the-badge&logo=pytorch)](https://pytorch.org)
[![NLP](https://img.shields.io/badge/NLP-LSTM-orange?style=for-the-badge)](https://en.wikipedia.org/wiki/Long_short-term_memory)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

> 🔮 Predict the **next word** in a sentence using **LSTM (Long Short-Term Memory)** neural networks — built with PyTorch!

---

## 🚀 What is this?

A **deep learning NLP model** that learns patterns from text and predicts what word comes next. Great for:
- 📝 Auto-complete systems
- 💬 Chatbot response generation
- 📖 Language modeling research
- 🎓 Learning LSTM / RNN architectures

---

## ✨ Features

- 🧠 LSTM-based sequence model
- 🔥 Built with PyTorch
- 📊 Trained on custom text data
- 🎯 Top-K word prediction
- 📈 Training loss visualization

---

## 📦 Installation

```bash
git clone https://github.com/Pokemon455/LSTM-prediction-next-word-.git
cd LSTM-prediction-next-word-
pip install torch numpy
```

## 🏃 Usage

```python
from model import NextWordPredictor

predictor = NextWordPredictor()
predictor.load("model.pth")

result = predictor.predict("The weather today is")
print(result)  # → "sunny"
```

---

## 🏗️ Model Architecture

```
Input Text → Tokenizer → Embedding Layer → LSTM Layers → Linear → Softmax → Next Word
```

| Layer | Size |
|-------|------|
| Embedding | 128 |
| LSTM Hidden | 256 |
| LSTM Layers | 2 |
| Vocabulary | Dynamic |

---

## 📊 Results

The model achieves strong perplexity scores on held-out test data with proper training.

---

## 🌟 Star this repo if it helped you!

[![GitHub stars](https://img.shields.io/github/stars/Pokemon455/LSTM-prediction-next-word-?style=social)](https://github.com/Pokemon455/LSTM-prediction-next-word-/stargazers)

---

## 👨‍💻 Author

**Arbaz** — AI/ML Developer
- GitHub: [@Pokemon455](https://github.com/Pokemon455)
- Bio: 🤖 AI/ML Developer | FastMCP • LangChain • PyTorch
