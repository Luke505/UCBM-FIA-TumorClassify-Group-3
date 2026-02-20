# UCBM-FIA TumorClassify (Group 3)

## Project Overview

A complete **machine learning pipeline for binary tumor classification** (benign vs. malignant) using a **k-Nearest Neighbors (k-NN) classifier** built from scratch, without external ML libraries like scikit-learn.

**Developed by Group 3** ([Luca Ciampa](https://github.com/Luke505), [Alfonso Ciampa](https://github.com/AlfonsoCiampa), [Even Ciampa](https://github.com/EvenCiampa)) for the **Fundamentals of Artificial Intelligence (FIA)** course at **Università Campus Bio-Medico di Roma (UCBM)**.

### Key Features

- **Object-Oriented Model** - TumorSample, TumorFeatures, TumorDataset classes
- **Automatic Data Validation** - Invalid samples filtered at load time
- **k-NN classifier** implemented from scratch (no scikit-learn)
- **Flexible Column Mapping** - Custom aliases for non-standard column names
- Supports 5 data formats (CSV, TXT, JSON, XLSX, TSV)
- 8 validation strategies (Holdout, Random Subsampling, K-Fold, LOOCV, Leave-p-out, Stratified K-Fold, Stratified Shuffle Split, Bootstrap)
- 6 evaluation metrics (Accuracy, Error Rate, Sensitivity, Specificity, Geometric Mean, AUC)
- User can select which metrics to validate via `--metrics`
- 188 comprehensive tests (100% passing)
- Design patterns (Factory, Strategy)
- Docker support

---

## Quick Statistics

| Component         | Details                             |
|-------------------|-------------------------------------|
| **Source Files**  | 13 Python modules                   |
| **Test Coverage** | 188 tests (100% passing)            |
| **Accuracy**      | 93-97% across datasets              |
| **Dependencies**  | numpy, pandas, matplotlib, openpyxl |

---

## Dataset

The dataset contains information about tumor cell characteristics and corresponding class labels.

### Features (Independent Variables)

Each sample includes nine numerical features (range 1-10):

1. **Clump Thickness** - Thickness of cell clump
2. **Uniformity of Cell Size** - Consistency in cell size
3. **Uniformity of Cell Shape** - Consistency in cell shape
4. **Marginal Adhesion** - Cell adhesion at margins
5. **Single Epithelial Cell Size** - Size of epithelial cells
6. **Bare Nuclei** - Presence of bare nuclei
7. **Bland Chromatin** - Chromatin texture
8. **Normal Nucleoli** - Nucleoli characteristics
9. **Mitoses** - Mitotic activity

### Class Labels

- `2` - **Benign tumor** (non-cancerous)
- `4` - **Malignant tumor** (cancerous)

### Supported Data Formats

The project handles multiple file formats with robust error handling:

- **CSV** (`.csv`) - Comma-separated values
- **TXT** (`.txt`) - Tab-separated text files
- **JSON** (`.json`) - JSON formatted data
- **Excel** (`.xlsx`) - Excel spreadsheets
- **TSV** (`.tsv`) - Tab-separated values

All loaders handle:

- **Automatic validation** - Invalid samples filtered at load time
- **Duplicate IDs** - Automatically ignored
- **Invalid values** - Samples with missing or invalid features rejected
- Multiple column name variations (built-in aliases)
- Text and numeric class labels
- Empty rows and invalid entries

---

## Model Architecture

### Object-Oriented Data Model

The project uses a clean object-oriented approach with three core classes:

**`TumorFeatures`** - Represents the 9 tumor features (integers 1-10):

- All features validated at creation
- Provides `to_array()` for ML algorithms
- Supports flexible dictionary input via `from_dict()`

**`TumorSample`** - Complete tumor sample:

- `id`: int (unique identifier)
- `features`: TumorFeatures object
- `tumor_class`: int (2 for benign, 4 for malignant)
- Properties: `is_malignant`, `is_benign`

**`TumorDataset`** - Collection of samples:

- Automatic duplicate ID filtering
- Class distribution tracking
- Array conversion for ML algorithms via `to_arrays()`
- Filtering and splitting capabilities

### Automatic Data Validation

Samples are automatically **rejected** during loading if:

- Duplicate ID (only first occurrence kept)
- Invalid class (not 2 or 4)
- Missing feature values
- Invalid feature values (not integers 1-10)

### k-NN Classifier

- **Algorithm:** k-Nearest Neighbors (implemented from scratch)
- **Distance Metric:** Euclidean distance
- **Tie-breaking:** Random selection among most frequent labels
- **Implementation:** No external ML libraries (only numpy for arrays)

### Data Preprocessing

- **Data Validation:** Automatic at load time (duplicates, invalid classes, invalid features)
- **Feature Normalization:** Min-max scaling to [0, 1] range
- **Label Validation:** Ensures labels are 2 (benign) or 4 (malignant)
- **Column Mapping:** Flexible aliases for non-standard column names

---

## Project Structure

```
.
├── docs/                # Project documentation (PDFs)
├── src/                 # Source code
│   ├── main.py          # CLI application
│   ├── model/           # Data models (TumorSample, TumorFeatures, TumorDataset)
│   ├── classifier/      # k-NN implementation
│   ├── evaluation/      # Validation strategies
│   ├── metrics/         # Evaluation metrics
│   ├── utils/           # Data loaders & preprocessing
│   └── visualization/   # Plotting functions
├── tests/               # Test suite (188 tests)
├── tests_data/          # Test datasets (5 formats)
├── data/                # Input data directory
├── results/             # Output results directory
├── Dockerfile           # Docker configuration
├── docker-compose.yml   # Docker Compose configuration
└── requirements.txt     # Python dependencies
```

---

## Installation

### Prerequisites

- Python 3.11 or higher
- (Optional) Docker Desktop for containerized execution

### Setup

```bash
# Clone repository
git clone https://github.com/Luke505/UCBM-FIA-TumorClassify-Group-3
cd UCBM-FIA-TumorClassify-Group-3

# Create virtual environment
python -m venv env
source env/bin/activate  # Linux/macOS
# or: env\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

### Basic Usage

```bash
# Run with default settings (holdout validation, k=3)
python -m src.main --data tests_data/version_1.csv

# Run with K-Fold cross-validation
python -m src.main --data tests_data/version_1.csv --k 5 --strategy kfold --K 10

# Run with verbose output
python -m src.main --data tests_data/version_1.csv --verbose
```

### Command-Line Options

```bash
python -m src.main --help
```

**Available options:**

| Option             | Description                                                 | Default     |
|--------------------|-------------------------------------------------------------|-------------|
| `--data, -d`       | Input data file path                                        | Auto-detect |
| `--aliases`        | Column name aliases (e.g., "id:patient_id,class:diagnosis") | None        |
| `--k`              | Number of neighbors for k-NN                                | 3           |
| `--k-values`       | Compare multiple k values (e.g., "1,3,5,7,9")               | None        |
| `--strategy, -s`   | Validation strategy (see below)                             | holdout     |
| `--K`              | Number of folds/splits/iterations                           | 5           |
| `--test-size`      | Test set proportion (for holdout/random subsampling)        | 0.3         |
| `--p`              | Number of samples to leave out (for leave-p-out)            | 2           |
| `--metrics, -m`    | Metrics to validate (see below)                             | all         |
| `--output-dir, -o` | Output directory                                            | results     |
| `--no-plots`       | Disable plot generation                                     | False       |
| `--no-subfolder`   | Save outputs directly to output-dir (no timestamp)          | False       |
| `--random-state`   | Random seed for reproducibility                             | 42          |
| `--verbose, -v`    | Enable verbose output                                       | False       |

**Note:** `k` and `K` are two distinct parameters: `k` is the number of neighbors for the k-NN classifier, while `K` is the number of experiments/folds for the validation strategy.

### Available Strategies

| Strategy               | CLI value             | Description                                      |
|------------------------|-----------------------|--------------------------------------------------|
| Holdout                | `holdout`             | Single train/test split                          |
| Random Subsampling (B) | `random_subsampling`  | Repeated random holdout (K iterations)           |
| K-Fold CV              | `kfold`               | K-Fold cross-validation                          |
| Leave-One-Out CV       | `loocv`               | Leave one sample out per fold                    |
| Leave-p-Out CV (C)     | `lpocv`               | Leave p samples out per fold                     |
| Stratified K-Fold      | `stratified`          | K-Fold preserving class proportions              |
| Stratified Shuffle     | `stratified_shuffle`  | Stratified repeated random splits                |
| Bootstrap              | `bootstrap`           | Bootstrap resampling with out-of-bag test        |

### Available Metrics

The user can select which metrics to validate using `--metrics`:

| Metric          | CLI value        |
|-----------------|------------------|
| Accuracy Rate   | `accuracy`       |
| Error Rate      | `error_rate`     |
| Sensitivity     | `sensitivity`    |
| Specificity     | `specificity`    |
| Geometric Mean  | `geometric_mean` |
| Area Under Curve| `auc`            |
| All of the above| `all` (default)  |

Example: `--metrics accuracy sensitivity auc` to show only those three.

### Examples

```bash
# Holdout validation (default) with k=5
python -m src.main --data tests_data/version_1.csv --k 5 --strategy holdout --test-size 0.3

# B: Random Subsampling with K=10 experiments
python -m src.main --data tests_data/version_1.csv --k 3 --strategy random_subsampling --K 10

# C: Leave-p-out Cross Validation with p=2 and K=100 max splits
python -m src.main --data tests_data/version_1.csv --k 3 --strategy lpocv --p 2 --K 100

# K-Fold cross-validation with 5 folds
python -m src.main --data tests_data/version_1.csv --k 5 --strategy kfold --K 5

# Leave-One-Out
python -m src.main --data tests_data/version_3.txt --k 3 --strategy loocv

# Bootstrap with 50 iterations
python -m src.main --data tests_data/version_5.tsv --k 7 --strategy bootstrap --K 50

# Validate only Accuracy and AUC
python -m src.main --data tests_data/version_1.csv --k 5 --strategy kfold --K 5 --metrics accuracy auc

# Compare multiple k values to find optimal k
python -m src.main --data tests_data/version_1.csv --k-values "1,3,5,7,9,11" --strategy kfold --K 10
```

### Custom Column Names

If your data uses different column names, use the `--aliases` parameter:

```bash
# Format: "standard_name:your_column_name,..."
python -m src.main --data mydata.csv --aliases "id:patient_id,clump_thickness:thickness_measure,class:diagnosis_code"

# Or use JSON format
python -m src.main --data mydata.csv --aliases '{"id":"patient_id","class":"diagnosis_code"}'
```

**Supported alias keys:**

- `id` - Sample identifier
- `clump_thickness`, `uniformity_cell_size`, `uniformity_cell_shape`
- `marginal_adhesion`, `single_epithelial_cell_size`, `bare_nuclei`
- `bland_chromatin`, `normal_nucleoli`, `mitoses`
- `class` - Tumor classification

---

## Docker

### Quick Start

```bash
# Prepare data
mkdir -p data
cp tests_data/version_1.csv data/

# Run with Docker Compose
docker-compose up --build
```

The `docker-compose.yml` file mounts two volumes:

- `./data:/app/data:ro` - Input data (read-only)
- `./results:/app/results` - Output results (read-write)

Results are automatically saved to your local `results/` folder.
To change parameters, edit the `command` section in `docker-compose.yml`:

```yaml
# Standard mode
command: [ "python", "-m", "src.main", "--data", "data/version_1.csv", "--k", "5", "--strategy", "kfold", "--K", "3" ]
```

```yaml
# Or compare multiple k values
command: [ "python", "-m", "src.main", "--data", "data/version_1.csv", "--k-values", "1,3,5,7,9", "--strategy", "kfold" ]
```

### Manual Docker

```bash
# Build image
docker build -t ucbm-fia-tumorclassify-group-3 .

# Run container
docker run --rm \
  -v $(pwd)/data:/app/data:ro \
  -v $(pwd)/results:/app/results \
  ucbm-fia-tumorclassify-group-3 \
  python -m src.main --data data/version_1.csv --k 5 --strategy kfold --K 3
```

---

## Results and Output

### Performance Across Datasets

| Dataset        | Format | Accuracy |
|----------------|--------|----------|
| version_1.csv  | CSV    | 98.75%   |
| version_2.xlsx | Excel  | 93.69%   |
| version_3.txt  | TXT    | 97.06%   |
| version_4.json | JSON   | 98.32%   |
| version_5.tsv  | TSV    | 97.03%   |

### Example Output (K-Fold, k=5)

```
Average Metrics:
  Accuracy:           0.9683 +/- 0.0140
  Error Rate:         0.0317 +/- 0.0140
  Sensitivity:        0.9529 +/- 0.0292
  Specificity:        0.9748 +/- 0.0190
  Geometric Mean:     0.9636 +/- 0.0166
  Auc:                0.9902 +/- 0.0084

Confusion Matrix:
  True Negatives (TN):  339
  False Positives (FP): 9
  False Negatives (FN): 8
  True Positives (TP):  180
```

### Output Files

Results are saved to `results/` directory with optional subfolder organization:

**Standard mode:** Single timestamped subfolder (e.g., `results/version_1_20260105_085526/`)
**K-values mode (`--k-values`):** Subfolders per k value (`k1/`, `k3/`, etc.) + comparison plots in parent

Use `--no-subfolder` to save directly to `results/`.

**Data Files:**

- `*_metrics.csv` - Per-fold metrics
- `*_summary.csv` - Summary statistics
- `k_comparison_results.csv` - K-values comparison (only in k-values mode)

**Visualization Plots:**

- `*_confusion_matrix.png` - Binary classification confusion matrix with TP/TN/FP/FN labels
- `*_roc_curve.png` - ROC curve with AUC score and diagonal reference line
- `*_metrics_comparison.png` - Bar chart comparing 5 key metrics (Accuracy, Error Rate, Sensitivity, Specificity, Geometric Mean)
- `*_accuracy_distribution.png` - Boxplot showing accuracy variability across folds (only for multi-fold strategies)
- `accuracy_vs_k.png` - Accuracy trend across k values (k-values mode only)
- `error_rate_vs_k.png` - Error rate trend across k values (k-values mode only)

### Interpreting Results

- **Accuracy > 0.9** indicates strong classification performance.
- **Sensitivity** measures how well the model detects malignant tumors (true positive rate).
- **Specificity** measures how well the model identifies benign tumors (true negative rate).
- **AUC > 0.9** indicates excellent discrimination between classes.
- **ROC curve** above the diagonal line means the classifier is better than random guessing.
- **Confusion Matrix** shows the breakdown of correct and incorrect predictions.

---

## Design

### Architecture

- **Object-Oriented Design** - TumorSample, TumorFeatures, TumorDataset models
- **Automatic Validation** - Data validated at load time
- **Factory Pattern** - Data loader creation based on file format (`DataLoaderFactory`)
- **Strategy Pattern** - Flexible validation strategy selection (`ValidationStrategyFactory`)
- **Modular** - Independent, reusable components
- **Type Safety** - Proper type hints throughout

### Key Components

1. **Data Model** (`src/model/`) - TumorSample, TumorFeatures, TumorDataset classes
2. **KNNClassifier** (`src/classifier/`) - k-NN implementation from scratch
3. **DataLoaderFactory** (`src/utils/`) - Creates loaders for different file formats
4. **ValidationStrategyFactory** (`src/evaluation/`) - Creates validation strategies
5. **Metrics Module** (`src/metrics/`) - All evaluation metrics implemented from scratch
6. **Visualization** (`src/visualization/`) - Plotting and result export

### Data Flow

```
Input File --> DataLoader --> TumorDataset (validated samples)
                                  |
                             to_arrays() --> numpy arrays
                                  |
                          Normalization --> [0,1] range
                                  |
                      Validation Strategy --> Train/Test splits
                                  |
                        k-NN Classifier --> Predictions
                                  |
                             Metrics --> Evaluation results
                                  |
                         Visualization --> Plots & CSV files
```

---

## Testing

Run the test suite:

```bash
source env/bin/activate
python -m unittest discover tests -v
```

The project includes **188 tests** covering:

- **Classifier tests** - k-NN prediction, distance calculation, edge cases
- **Data loader tests** - All 5 file formats, validation, column mapping
- **Metrics tests** - All 6 metrics, confusion matrix, edge cases
- **Validation strategy tests** - All 8 strategies, splits, reproducibility
- **Preprocessing tests** - Feature normalization
- **Visualization tests** - Plot generation, file saving
- **Integration tests** - Full pipeline with real data
- **Edge case tests** - Empty data, imbalanced classes, model validation

---

## Troubleshooting

### Common Issues

#### Issue: Module not found error

**Solution:** Ensure virtual environment is activated and dependencies are installed:

```bash
source env/bin/activate  # or env\Scripts\activate on Windows
pip install -r requirements.txt
```

#### Issue: No data file found

**Solution:** Specify data file explicitly:

```bash
python -m src.main --data tests_data/version_1.csv
```

#### Issue: Docker daemon not running

**Solution:** Start Docker Desktop and wait for it to initialize.

#### Issue: Permission denied on results folder

**Solution:** Create and set permissions:

```bash
mkdir -p results
chmod -R 755 results
```

---

## Authors

**Group 3:**

- [Luca Ciampa](https://github.com/Luke505)
- [Alfonso Ciampa](https://github.com/AlfonsoCiampa)
- [Even Ciampa](https://github.com/EvenCiampa)

**Course:** Fundamentals of Artificial Intelligence (FIA)
**Institution:** Università Campus Bio-Medico di Roma (UCBM)
**Year:** 2025-2026

---

## Additional Resources

- [Project Requirements](docs/Project.pdf) - Official FIA project requirements
- [Development Guidelines](docs/Guida_allo_sviluppo_di_progetti.pdf) - FIA development guidelines
- [Confusion Matrix](https://en.wikipedia.org/wiki/Confusion_matrix)
- [ROC Curve](https://en.wikipedia.org/wiki/Receiver_operating_characteristic)
- [k-NN Algorithm](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm)

---

**For questions or support, please open an issue on [GitHub](https://github.com/Luke505/UCBM-FIA-TumorClassify-Group-3/issues).**
