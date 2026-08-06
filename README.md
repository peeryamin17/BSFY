# KATHE 2026: English-to-Kashmiri Translation Model

**Team**: Yameen Javid, Salik Javid, Mir Behzad, Farhaan Pala
**Competition**: [KATHE 2026 on Kaggle](https://www.kaggle.com/c/kathe-2026)
**Leaderboard Score**: TBD

Fine-tuned neural machine translation model for **English → Kashmiri (kas_Arab)**,
scored on the **geometric mean of chrF++ and BLEU**.

## Model

- **Base**: `ai4bharat/indictrans2-en-indic-1B` — IndicTrans2 gives the best
  Kashmiri coverage among pretrained models.
- **Method**: LoRA fine-tuning with cached, parallel tokenization and
  gradient checkpointing — trainable parameters are a small fraction of the
  base model, so it fits comfortably on an 8GB GPU.
- **Fine-tuned on**: BPCC Kashmiri subset
  (`ai4bharat/BPCC` · `bpcc-seed-latest/kas_Arab.tsv`, ~28 MB).
- **Metric**: `geo_mean = sqrt(chrF++ × BLEU)`.

## Quick Start

### 1. Installation

```bash
git clone https://github.com/yaminjavid/kathe-2026.git
cd kathe-2026
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

> The BPCC dataset and IndicTrans2 weights are gated on HuggingFace, so log in
> once and accept the dataset/model terms first:
>
> ```bash
> huggingface-cli login
> ```

### 2. Download data and build splits

```bash
python data/download_data.py          # pulls data/raw/kas_Arab.tsv
python data/train_val_split.py        # writes data/{train,val,test}.csv
```

### 3. Fine-tune

```bash
python model/train.py --config model/config.yaml
```

The best checkpoint (highest val geo-mean) is saved to `outputs/models/`.

### 4. Evaluate on the held-out split

```bash
python model/evaluate.py --checkpoint outputs/models/nllb-baseline
```

### 5. Make a Kaggle submission

```bash
python model/inference.py \
    --checkpoint outputs/models/nllb-baseline \
    --test data/test.csv \
    --out outputs/submission.csv
```

### 6. Translate in code

```python
from model.inference import translate

texts = ["Hello, how are you?", "What is your name?"]
predictions = translate(texts, checkpoint="outputs/models/nllb-baseline")
for pred in predictions:
    print(pred)
```

## Repository Layout

```
kathe-2026-english-kashmiri/
├── README.md
├── LICENSE
├── requirements.txt
├── data/
│   ├── download_data.py          # fetch BPCC kas_Arab.tsv from HF
│   ├── train_val_split.py        # create train/val/test splits
│   ├── train.csv                 # (generated)
│   ├── val.csv                   # (generated)
│   └── test.csv                  # (generated)
├── model/
│   ├── config.yaml               # all hyperparameters live here
│   ├── train.py                  # fine-tuning
│   ├── evaluate.py               # chrF++ / BLEU / geo-mean
│   └── inference.py              # generate predictions + helpers
├── notebooks/
│   └── experiment_log.ipynb      # exploration notes
└── outputs/
    └── submission.csv            # final Kaggle submission
```

## Results

| Metric   | Score |
|----------|-------|
| chrF++   | XX.XX |
| BLEU     | XX.XX |
| Geo-Mean | XX.XX |

## License

MIT License — see [LICENSE](LICENSE).

## Authors

- Yameen Javid (yaminjavid18@gmail.com)
- Salik Javid (Saliklone624@gmail.com)
- Mir Behzad
- Farhaan Pala
