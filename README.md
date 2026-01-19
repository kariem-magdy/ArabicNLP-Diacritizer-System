# Arabic Text Diacritization System

> An end-to-end NLP pipeline for automatic Arabic diacritization (tashkeel) using BiLSTM-CRF and Transformer-based models.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Overview

This project implements a comprehensive Natural Language Processing (NLP) pipeline for automatic Arabic diacritization.  The system takes undiacritized Arabic text as input and produces fully diacritized text with correct vowel marks (harakat), sukun, and shadda.

The project provides two state-of-the-art approaches:
1. **BiLSTM-CRF**: A feature-rich sequence labeling model combining character embeddings, word embeddings, and optional sentence-level features (BoW, TF-IDF, FastText)
2. **Transformer Fine-Tuning**:  Fine-tuned AraBERT model for token classification with automatic subword alignment

Both models can be used independently or combined through an ensemble predictor for improved accuracy.

---

## Problem Statement

Arabic script is inherently ambiguous without diacritical marks (tashkeel). The same consonantal skeleton can represent multiple words with entirely different meanings: 

| Undiacritized | Possible Diacritized Forms | Meanings |
|---------------|---------------------------|----------|
| كتب | كَتَبَ / كُتُب / كُتِبَ | he wrote / books / was written |
| علم | عَلِمَ / عِلْم / عَلَم | he knew / knowledge / flag |

### Sources of Ambiguity

1. **Lexical Ambiguity**: The same consonant sequence can form different words
2. **Morphological Ambiguity**: Arabic's rich morphology means one root can derive many forms
3. **Syntactic Ambiguity**: Case endings (i'rab) depend on grammatical role in the sentence
4. **Contextual Dependency**:  Correct diacritization often requires understanding the full sentence context

### Diacritical Marks in Arabic

| Mark | Name | Unicode | Description |
|------|------|---------|-------------|
| َ | Fatha | U+064E | Short /a/ vowel |
| ُ | Damma | U+064F | Short /u/ vowel |
| ِ | Kasra | U+0650 | Short /i/ vowel |
| ْ | Sukun | U+0652 | Absence of vowel |
| ّ | Shadda | U+0651 | Consonant gemination |
| ً | Tanwin Fath | U+064B | Nunation with fatha |
| ٌ | Tanwin Damm | U+064C | Nunation with damma |
| ٍ | Tanwin Kasr | U+064D | Nunation with kasra |

The system also handles **compound marks** (e.g., Shadda + Fatha:  ّ + َ) as single labels.

---

## System Pipeline / End-to-End Workflow

### High-Level Pipeline Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         TRAINING PIPELINE                                    │
└─────────────────────────────────────────────────────────────────────────────┘

    Raw Diacritized Text (train. txt)
              │
              ▼
    ┌─────────────────────┐
    │   Text Cleaning     │  Remove URLs, emails, zero-width chars
    └─────────────────────┘
              │
              ▼
    ┌─────────────────────┐
    │  Label Extraction   │  Separate base chars from diacritics
    └─────────────────────┘
              │
              ▼
    ┌─────────────────────┐
    │   Normalization     │  Hamza normalization, tatweel removal
    └─────────────────────┘
              │
              ▼
    ┌─────────────────────┐
    │  Vocabulary Build   │  char2idx, label2idx, word2idx
    └─────────────────────┘
              │
              ▼
    ┌─────────────────────┐
    │  Feature Fitting    │  BoW/TF-IDF vectorizers (optional)
    └─────────────────────┘
              │
              ▼
    ┌─────────────────────┐
    │   Model Training    │  BiLSTM-CRF or Transformer
    └─────────────────────┘
              │
              ▼
    ┌────────���────────────┐
    │    Evaluation       │  DER (Diacritic Error Rate)
    └─────────────────────┘
              │
              ▼
       Saved Model Checkpoint


┌─────────────────────────────────────────────────────────────────────────────┐
│                        INFERENCE PIPELINE                                    │
└─────────────────────────────────────────────────────────────────────────────┘

    Raw Undiacritized Text
              │
              ▼
    ┌─────────────────────┐
    │  Strip Diacritics   │  Remove any existing marks
    └─────────────────────┘
              │
              ▼
    ┌─────────────────────┐
    │   Normalization     │  Same as training
    └─────────────────────┘
              │
              ▼
    ┌─────────────────────┐
    │ Feature Extraction  │  Chars + Words + BoW/TF-IDF
    └─────────────────────┘
              │
              ▼
    ┌─────────────────────┐
    │   Model Forward     │  BiLSTM-CRF / Transformer / Ensemble
    └─────────────────────┘
              │
              ▼
    ┌─────────────────────┐
    │  Label Decoding     │  CRF decode / argmax
    └─────────────────────┘
              │
              ▼
    ┌─────────────────────┐
    │  Reconstruction     │  Merge chars + predicted diacritics
    └─────────────────────┘
              │
              ▼
    Fully Diacritized Arabic Text
```

---

### Step-by-Step Pipeline Explanation

#### Stage 1: Text Cleaning (`src/preprocess. py:: clean_line`)

**Purpose**: Remove noise and non-essential content from raw text.

**Input**: Raw text line (potentially with URLs, emails, control characters)

**Output**:  Cleaned text string

**Operations**:
- URL removal via regex (`https?://\S+`)
- Email removal via regex
- Zero-width character removal (U+200B-U+200F, FEFF)
- Control character filtering (Unicode category "C")
- Whitespace normalization (collapse multiple spaces)

**Why Necessary**: Arabic web text often contains mixed content.  Cleaning ensures the model focuses on actual Arabic text.

---

#### Stage 2: Label Extraction (`src/preprocess. py::extract_labels`)

**Purpose**: Separate base characters from their diacritical marks while preserving the alignment.

**Input**:  Diacritized Arabic text (e.g., `كَتَبَ`)

**Output**: 
- Base characters string:  `كتب`
- Labels list: `['َ', 'َ', 'َ']`

**Algorithm**:
1. Iterate through each character
2. If character is a combining mark (Unicode category "M"), append to pending marks
3. If character is a base character, attach pending marks to previous character as label
4. Handle compound diacritics (Shadda + vowel) as single labels

**Why Necessary**: Arabic diacritics are Unicode combining marks that attach to preceding characters. The model predicts diacritics as labels for each base character.

---

#### Stage 3: Normalization (`src/preprocess. py::normalize_base_and_labels`)

**Purpose**: Standardize text variations and reduce vocabulary size.

**Input**: Base text + labels list

**Output**:  Normalized base text + synchronized labels

**Normalization Options** (configurable):
| Option | Description | Default |
|--------|-------------|---------|
| `normalize_hamza` | Normalize أ/إ/آ → ا | True |
| `remove_tatweel` | Remove kashida (ـ) | True |
| `lower_latin` | Lowercase Latin chars | True |
| `remove_punctuation` | Remove punctuation | False |

**Why Necessary**: 
- Hamza normalization reduces character variants
- Synchronized normalization prevents label-character misalignment
- Consistent preprocessing between training and inference is critical

---

#### Stage 4: Vocabulary Building (`build_vocab.py`)

**Purpose**: Create character-to-index and label-to-index mappings.

**Input**: Cleaned, normalized training corpus

**Output** (saved to `outputs/processed/`):
- `char2idx. json`: Character vocabulary (includes `<PAD>`, `<UNK>`)
- `label2idx.json`: Diacritic label vocabulary (includes `_` for no diacritic)
- `word2idx.json`: Word vocabulary (if word embeddings enabled)
- `idx2char.json`, `idx2label.json`: Inverse mappings

**Why Necessary**: Neural networks require numerical inputs. Vocabularies enable consistent encoding.

---

#### Stage 5: Feature Extraction (`src/features.py`, `src/data/dataset.py`)

**Purpose**: Convert text into numerical feature vectors.

**Features Extracted**: 

| Feature | Dimension | Source | Description |
|---------|-----------|--------|-------------|
| Character Embeddings | 128 | Trainable | Per-character learned embeddings |
| Word Embeddings | 128 | Trainable | Per-word learned embeddings (aligned to chars) |
| FastText Embeddings | 300 | Pre-trained | Arabic Wikipedia word vectors (optional) |
| Bag of Words | 32 | Sentence-level | Projected BoW features (optional) |
| TF-IDF | 32 | Sentence-level | Projected TF-IDF features (optional) |

**Why Necessary**: 
- Character embeddings capture letter-level patterns
- Word embeddings provide lexical context
- Sentence-level features (BoW/TF-IDF) add global context for disambiguation

---

#### Stage 6: Model Forward Pass

##### BiLSTM-CRF Architecture (`src/models/bilstm_crf.py`)

```
Input Features (concatenated)
        │
        ▼
┌───────────────────┐
│   BiLSTM Layer    │  3 layers, hidden=256, bidirectional
│   (Contextual)    │
└───────────────────┘
        │
        ▼
┌───────────────────┐
│   Linear Layer    │  Project to num_labels
└───────────────────┘
        │
        ▼
┌───────────────────┐
│    CRF Layer      │  Model label transitions
└───────────────────┘
        │
        ▼
    Predicted Labels
```

**Why CRF**: The CRF layer models transition probabilities between diacritics, preventing invalid sequences (e.g., two consecutive sukuns).

##### Transformer Architecture

- Model:  `aubmindlab/bert-base-arabertv02`
- Task: Token classification (character-level)
- Subword alignment: First subtoken of each character receives the label

---

#### Stage 7: CRF Decoding (`src/models/crf_layer.py`)

**Purpose**: Decode the most likely sequence of labels using Viterbi algorithm. 

**Input**:  Emission scores from BiLSTM + CRF transition matrix

**Output**:  Optimal label sequence

**Why Necessary**: CRF decoding considers the full sequence, not just per-position argmax, leading to globally consistent predictions.

---

#### Stage 8: Reconstruction (Inference Scripts)

**Purpose**: Merge predicted diacritics with base characters. 

**Input**: 
- Base characters: `ذهب الطالب`
- Predicted labels: `['َ', 'َ', 'َ', '_', 'ْ', 'ُ', ...]`

**Output**: `ذَهَبَ الطَّالِبُ`

**Algorithm**:
1. Iterate through base characters
2.  Append character + corresponding diacritic (if not `_`)
3. Preserve spaces

---

### Context Handling

The system handles context at **sentence level**:
- **BiLSTM**:  Bidirectional processing captures both left and right context
- **CRF**: Transition matrix models diacritic sequence patterns
- **Transformer**: Self-attention attends to all positions in the sentence
- **Sentence Features**: BoW/TF-IDF provide global topical context

**Processing Level**: The system processes text at **sentence level** (line-by-line), not word-level.  This enables contextual disambiguation.

---

### Language Support

| Variant | Supported | Notes |
|---------|-----------|-------|
| Modern Standard Arabic (MSA) | ✅ Yes | Primary focus |
| Classical Arabic | ⚠️ Partial | Depends on training data |
| Dialectal Arabic | ❌ No | Not trained on dialects |

---

## Key Features

### 1. BiLSTM-CRF Architecture
- Multi-layer bidirectional LSTM (default:  3 layers)
- CRF layer for modeling label transitions
- Configurable feature combination: 
  - Character embeddings (always on)
  - Trainable word embeddings
  - Pre-trained FastText embeddings
  - Bag-of-Words sentence features
  - TF-IDF sentence features

### 2. Transformer Fine-Tuning
- Fine-tunes AraBERT (`aubmindlab/bert-base-arabertv02`)
- Automatic subword-to-character alignment
- HuggingFace `Trainer` API integration

### 3. Ensemble Prediction
- Combines BiLSTM-CRF and Transformer predictions
- Configurable weighting between models
- Improved robustness through model diversity

### 4. Production-Ready Components
- Streamlit web application for interactive demos
- Config-driven architecture
- Modular, extensible codebase

---

## Models & NLP Techniques

### BiLSTM-CRF

| Component | Configuration |
|-----------|---------------|
| Character Embedding | 128 dimensions, trainable |
| Word Embedding | 128 dimensions, trainable (optional) |
| FastText | 300 dimensions, frozen (optional) |
| BiLSTM | 3 layers, 256 hidden units, 0. 5 dropout |
| CRF | Full transition matrix, Viterbi decoding |

**Loss Function**: Negative log-likelihood from CRF

### Transformer

| Component | Configuration |
|-----------|---------------|
| Base Model | `aubmindlab/bert-base-arabertv02` |
| Task Head | Token classification |
| Max Length | 512 tokens |
| Learning Rate | 3e-5 |

---

## Dataset(s)

### Expected Format

Training and validation files should contain **diacritized Arabic text**, one sentence per line: 

```
ذَهَبَ الطَّالِبُ إِلَى المَدْرَسَةِ
كَتَبَ الأُسْتَاذُ الدَّرْسَ عَلَى السَّبُّورَةِ
```

### Data Directory Structure

```
data/
├── train.txt              # Training data (diacritized)
├── val. txt                # Validation data (diacritized)
├── test. txt               # Test data (undiacritized for inference)
└── fasttext_wiki.ar.vec   # Optional:  FastText Arabic vectors
```

---

## Training & Evaluation

### Evaluation Metric:  Diacritic Error Rate (DER)

```
DER = (Substitutions + Insertions + Deletions) / Total Reference Characters × 100%
```

The DER metric implemented in `src/eval/metrics.py`:
- Counts character-level diacritic mismatches
- Penalizes length mismatches (insertions/deletions)
- Lower is better (0% = perfect)

### Training Workflow

1. **Vocabulary Generation**:  `python build_vocab.py`
2. **BiLSTM Training**: `python -m src.train. train_bilstm`
3. **Transformer Training**: `python -m src.train.train_transformer`

---

## Installation & Setup

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended)
- 8GB+ RAM

### Installation Steps

```bash
# 1. Clone the repository
git clone https://github.com/kariem-magdy/ArabicNLP-Diacritizer-System.git
cd ArabicNLP-Diacritizer-System

# 2. Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or:  venv\Scripts\activate  # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

### Dependencies

```
torch>=2.0.0
transformers>=4.30.0
datasets>=2.11.0
pytorch-crf>=0.7.2
flask>=2.2.0
tqdm
scikit-learn
sentencepiece
regex
pyarabic
accelerate
tensorboard
```

---

## Usage Examples

### 1. Build Vocabularies (Required First)

```bash
python build_vocab.py
```

This generates:
- Character and label vocabularies
- Word vocabulary (if enabled)
- BoW/TF-IDF vectorizers (if enabled)

### 2. Train BiLSTM-CRF Model

```bash
python -m src.train.train_bilstm
```

Output:  `outputs/models/best_bilstm. pt`

### 3. Train Transformer Model

```bash
python -m src. train.train_transformer
```

Output: `outputs/models/best_transformer/`

### 4. Run Inference (CLI)

```bash
python -m src. infer.infer
```

Modify the test sentence inside `src/infer/infer.py`.

### 5. Run Web Demo

```bash
streamlit run src/app/app.py
```

Access at: `http://localhost:8501`

### 6. Generate Competition Submission

```bash
python generate_submission.py
```

---

## Example Input → Output

### Input (Undiacritized)
```
ذهب الطالب الى المدرسة
```

### Output (Diacritized)
```
ذَهَبَ الطَّالِبُ إِلَى المَدْرَسَةِ
```

### Label Visualization

| Character | Predicted Label | Diacritized |
|-----------|-----------------|-------------|
| ذ | َ (fatha) | ذَ |
| ه | َ (fatha) | هَ |
| ب | َ (fatha) | بَ |
| ا | ّ (shadda) | ا |
| ل | ْ (sukun) | لْ |
| ط | َ (fatha) | طَ |
| ...  | ... | ... |

---

## Project Structure

```
ArabicNLP-Diacritizer-System/
├── build_vocab.py              # Vocabulary & feature vectorizer builder
├── generate_submission.py      # Competition submission generator
├── requirements.txt            # Python dependencies
├── arabic_letters.pickle       # Allowed character set
├── diacritic2id.pickle         # Diacritic mapping
├── diacritics.pickle           # Diacritic list
│
├── data/
│   ├── train.txt               # Training data
│   ├── val.txt                 # Validation data
│   └── test.txt                # Test data
│
├── outputs/
│   ├── models/                 # Saved model checkpoints
│   │   ├── best_bilstm.pt
│   │   └── best_transformer/
│   ├── logs/                   # Training logs
│   └── processed/              # Vocabularies & vectorizers
│       ├── char2idx.json
│       ├── label2idx.json
│       ├── word2idx.json
│       ├── bow.pkl
│       └── tfidf.pkl
│
├── scripts/
│   ├── run_train_bilstm.sh     # BiLSTM training script
│   └── run_train_transformer.sh # Transformer training script
│
└── src/
    ├── config.py               # Configuration & hyperparameters
    ├── preprocess.py           # Text cleaning & normalization
    ├── features.py             # BoW, TF-IDF, FastText features
    │
    ├── data/
    │   ├── dataset. py          # PyTorch Dataset
    │   └── collate. py          # Batch collation
    │
    ├── models/
    │   ├── bilstm_crf. py       # BiLSTM-CRF model
    │   ├── crf_layer.py        # CRF wrapper
    │   └── transformer_finetune.py  # Transformer classifier
    │
    ├── train/
    │   ├── train_bilstm.py     # BiLSTM training loop
    │   └── train_transformer. py # Transformer training
    │
    ├── infer/
    │   ├── infer.py            # BiLSTM inference
    │   ├── infer_transformer.py # Transformer inference
    │   └── infer_ensemble.py   # Ensemble inference
    │
    ├── eval/
    │   └── metrics.py          # DER calculation
    │
    ├── utils/
    │   └── checkpoints.py      # Model saving utilities
    │
    └── app/
        └── app.py              # Streamlit web application
```

---

## Configuration (`src/config.py`)

### Feature Flags

```python
# Enable/disable features
use_word_emb: bool = True      # Trainable word embeddings
use_fasttext: bool = False     # Pre-trained FastText
use_bow: bool = False          # Bag-of-Words features
use_tfidf: bool = False        # TF-IDF features
```

### Model Hyperparameters

```python
# BiLSTM
char_emb_dim: int = 128
word_emb_dim: int = 128
lstm_hidden:  int = 256
lstm_layers: int = 3
bilstm_dropout:  float = 0.5

# Training
batch_size:  int = 64
epochs: int = 20
lr: float = 1e-3

# Transformer
transformer_model_name: str = "aubmindlab/bert-base-arabertv02"
transformer_batch_size: int = 8
transformer_lr: float = 3e-5
transformer_epochs: int = 4
```

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'Add your feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

---

## License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.

---

```
