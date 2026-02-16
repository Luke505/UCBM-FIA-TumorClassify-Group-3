# UCBM-FIA TumorClassify — Group 3

Binary tumour classification using k-Nearest Neighbours (k-NN).

## Dataset

Tumour classification dataset: 9 integer features (1–10) per sample,
labelled as class 2 (benign) or class 4 (malignant).

## Requirements

- Python 3.11+
- numpy
- pandas

## Installation

```bash
pip install -r requirements.txt
```

## How to Run

```bash
python -m src.main --data tests_data/version_1.csv
python -m src.main --data tests_data/version_1.csv --strategy kfold --k 5 --K 10
```
