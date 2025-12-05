import os
from dataclasses import dataclass

@dataclass
class Config:
    # Data paths
    data_dir: str = os.path.join(os.path.dirname(__file__), '..', 'data')
    train_file: str = os.path.join(data_dir, 'train.txt')
    val_file: str = os.path.join(data_dir, 'val.txt')
    test_file: str = os.path.join(data_dir, 'test.txt')

    # Model & training
    device = "cuda" if (os.environ.get("CUDA_VISIBLE_DEVICES","") != "" or (hasattr(__import__('torch'),'cuda') and __import__('torch').cuda.is_available())) else "cpu"

    # Char embedding model defaults (BiLSTM)
    char_vocab_min_freq: int = 1
    char_emb_dim: int = 128
    lstm_hidden: int = 256
    lstm_layers: int = 2
    bilstm_dropout: float = 0.3
    batch_size: int = 32
    epochs: int = 20
    lr: float = 1e-3

    # Transformer defaults
    transformer_model_name: str = "aubmindlab/bert-base-arabertv02"  # change if you prefer
    transformer_batch_size: int = 8
    transformer_lr: float = 3e-5
    transformer_epochs: int = 4

    # Checkpoints / outputs
    outputs_dir: str = os.path.join(os.path.dirname(__file__), '..', 'outputs')
    models_dir: str = os.path.join(outputs_dir, 'models')
    logs_dir: str = os.path.join(outputs_dir, 'logs')

cfg = Config()
