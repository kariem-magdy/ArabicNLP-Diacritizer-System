# src/config.py
import os
from dataclasses import dataclass

@dataclass
class Config:
    # --- Paths (Your Original Paths) ---
    data_dir: str = os.path.join(os.path.dirname(__file__), '..', 'data')
    train_file: str = os.path.join(data_dir, 'train.txt')
    val_file: str = os.path.join(data_dir, 'val.txt')
    test_file: str = os.path.join(data_dir, 'test.txt')

    # --- Output Paths ---
    outputs_dir: str = os.path.join(os.path.dirname(__file__), '..', 'outputs')
    models_dir: str = os.path.join(outputs_dir, 'models')
    logs_dir: str = os.path.join(outputs_dir, 'logs')
    processed_dir: str = os.path.join(outputs_dir, 'processed')

    # --- System ---
    device = "cuda" if (os.environ.get("CUDA_VISIBLE_DEVICES", "") != "" or 
                       (hasattr(__import__('torch'), 'cuda') and __import__('torch').cuda.is_available())) else "cpu"

    # --- Model Defaults (BiLSTM) ---
    char_vocab_min_freq: int = 1
    char_emb_dim: int = 128
    lstm_hidden: int = 512
    lstm_layers: int = 3
    bilstm_dropout: float = 0.5
    
    # --- Training Defaults ---
    batch_size: int = 64
    epochs: int = 20
    lr: float = 1e-3

    # --- Transformer Defaults (Preserved) ---
    transformer_model_name: str = "aubmindlab/bert-base-arabertv02"
    transformer_batch_size: int = 8
    transformer_lr: float = 3e-5
    transformer_epochs: int = 4

    # ====================================================
    # NEW FEATURE FLAGS
    # ====================================================
    
    # 1. Trainable Word Embeddings
    use_word_emb: bool = True
    word_emb_dim: int = 300
    
    # 2. FastText Pre-trained Embeddings
    use_fasttext: bool = False
    fasttext_path: str = os.path.join(data_dir, "fasttext_wiki.ar.vec")
    fasttext_dim: int = 300
    
    # 3. Bag of Words (BoW)
    use_bow: bool = False
    bow_vocab_size: int = 2000
    bow_emb_dim: int = 32
    
    # 4. TF-IDF
    use_tfidf: bool = False
    tfidf_vocab_size: int = 2000
    tfidf_emb_dim: int = 32

cfg = Config()