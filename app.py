from pathlib import Path

import streamlit as st
import torch

from mlp_model import generate_text, load_artifacts

st.set_page_config(page_title="MLP Next Word Predictor", page_icon="🧠")
st.title("🧠 Simple MLP Next-Word Predictor")
st.caption("Model ko pehle `train.py` se train karo, phir yahan text generate karo.")

model_dir = Path(st.sidebar.text_input("Artifacts folder", value="artifacts"))
max_new_tokens = st.sidebar.slider("Max new words", min_value=1, max_value=50, value=15)
temperature = st.sidebar.slider("Temperature", min_value=0.2, max_value=2.0, value=1.0, step=0.1)

if not (model_dir / "model.pt").exists() or not (model_dir / "metadata.json").exists():
    st.warning("Model artifacts nahi mile. Pehle train run karo: `python train.py`")
    st.stop()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, vocab, inv_vocab, config = load_artifacts(model_dir, device)

seed_text = st.text_input("Starting text", "how are")

if st.button("Generate"):
    generated = generate_text(
        model=model,
        seed_text=seed_text,
        vocab=vocab,
        inv_vocab=inv_vocab,
        context_size=config.context_size,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        device=device,
    )
    st.subheader("Generated text")
    st.write(generated)
