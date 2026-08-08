# KATHE 2026: English-to-Kashmiri Translation Model

**Team**: Yameen Javid, Salik Javid, Mir Behzad, Farhaan Pala
**Competition**: [KATHE 2026 on Kaggle](https://www.kaggle.com/c/kathe-2026)
**Fine-tuned weights**: [peeryamin17/kathe-checkpoints on HuggingFace](https://huggingface.co/peeryamin17/kathe-checkpoints)

Fine-tuned neural machine translation model for **English → Kashmiri (kas_Arab)**,
scored on the **geometric mean of chrF++ and BLEU**.

## Model

- **Base**: `ai4bharat/indictrans2-en-indic-1B` — IndicTrans2 gives the best
  Kashmiri coverage among pretrained models.
- **Method**: LoRA fine-tuning (r=32, α=64) with cached tokenization and
  gradient checkpointing. Trainable parameters are ~3% of the base model, so
  it fits comfortably on an 8GB GPU.
- **Fine-tuned on**: BPCC Kashmiri subset
  (`ai4bharat/BPCC` · `bpcc-seed-latest/kas_Arab.tsv`, 94,891 sentence pairs).
- **Inference**: beam search (8 beams) with the official IndicProcessor for
  entity protection, plus Unicode NFC normalization and orthographic
  character alignment to the corpus's spelling conventions.
- **Metric**: `geo_mean = sqrt(chrF++ × BLEU)`.

## Results

Held-out validation (2,000-sentence sample, final model):

| Metric | Score |
|--------|-------|
| chrF++ | 81.41 |
| BLEU   | 76.70 |
| Geo-Mean | 79.02 |
| Precision | 61.93 |
| Recall | 51.44 |
| F1 | 55.40 |

## Quick Start

### 1. Installation

```bash
git clone https://github.com/peeryamin17/BSFY.git
cd BSFY
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

The best checkpoint (highest val geo-mean) is saved to `outputs/models/indic2-final`.

### 4. Evaluate on the held-out split

```bash
python model/evaluate.py --checkpoint outputs/models/indic2-final --max-samples 2000
```

### 5. Download the fine-tuned weights

```bash
huggingface-cli download peeryamin17/kathe-checkpoints \
    --local-dir outputs/models/indic2-final
```

### 6. Make a Kaggle submission

```bash
python model/inference.py \
    --checkpoint outputs/models/indic2-final \
    --test /kaggle/input/competitions/kathe-2026/englishdev.csv \
    --out outputs/submission.csv
```

The submission is a CSV with `ID` and `kashmiri_text` columns, as required
by the competition rules.

## Repository Layout

```
BSFY/
├── README.md
├── LICENSE
├── requirements.txt
├── data/
│   ├── download_data.py          # fetch BPCC kas_Arab.tsv from HF
│   ├── analyze_orthography.py    # learn corpus character conventions
│   ├── train_val_split.py        # create train/val/test splits (with checks)
│   ├── train.csv                 # (generated)
│   ├── val.csv                   # (generated)
│   └── test.csv                  # (generated)
├── model/
│   ├── config.yaml               # all hyperparameters live here
│   ├── train.py                  # fine-tuning (auto-backups checkpoints to HF)
│   ├── evaluate.py               # full metric report
│   ├── inference.py              # generate predictions
│   ├── make_submission.py        # one-shot submission generator
│   ├── metrics.py                # chrF++/BLEU/geo-mean + extra metrics
│   └── orthography.py            # NFC + character alignment
└── outputs/
    └── submission.csv            # final Kaggle submission
```

## License

MIT License — see [LICENSE](LICENSE).

## Authors

- Yameen Javid (yaminjavid18@gmail.com)
- Salik Javid
- Mir Behzad
- Farhaan Pala
