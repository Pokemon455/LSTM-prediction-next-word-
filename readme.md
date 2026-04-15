# Next Word Prediction with Simple MLP

Ye project ek **simple MLP (Multi-Layer Perceptron)** model train karta hai jo previous words dekh kar next word predict karta hai.

## Files
- `train.py` -> MLP model training script
- `mlp_model.py` -> model, vocab, inference helpers
- `app.py` -> Streamlit UI for text generation
- `Dataset.csv` -> question/answer dataset

## Setup
```bash
pip install -r requirements.txt
```

## 1) Train model
```bash
python train.py --data-path Dataset.csv --output-dir artifacts --epochs 8
```

Model artifacts save honge:
- `artifacts/model.pt`
- `artifacts/metadata.json`

## 2) Run UI
```bash
streamlit run app.py
```

Browser me prompt do, **Generate** dabao, aur model next words generate karega.

## Optional training args
```bash
python train.py \
  --context-size 3 \
  --embedding-dim 64 \
  --hidden-dim 128 \
  --batch-size 256 \
  --lr 0.001
```
