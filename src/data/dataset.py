# src/data/dataset.py
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Optional
from ..preprocess import parse_file_to_entries, UNK_TOKEN
from ..features import feature_mgr
from ..config import cfg

class DiacritizationDataset(Dataset):
    """
    PyTorch Dataset that delegates processing to src.preprocess but adds Feature Extraction.
    """
    def __init__(self, 
                 file_path: str, 
                 char2idx: Dict[str, int], 
                 label2idx: Dict[str, int],
                 word2idx: Optional[Dict[str, int]] = None,
                 normalize_options: Optional[Dict] = None):
        
        self.char2idx = char2idx
        self.label2idx = label2idx
        self.word2idx = word2idx
        
        # Load pre-trained features (BoW/TF-IDF)
        feature_mgr.load()
        
        # Default Normalization options matching filtering.py
        if normalize_options is None:
            normalize_options = {
                "normalize_hamza": True, 
                "remove_tatweel": False, 
                "lower_latin": True, 
                "remove_punctuation": True
            }

        # Delegate parsing to the robust preprocess module
        self.entries = parse_file_to_entries(
            file_path=file_path,
            char2idx=char2idx,
            label2idx=label2idx,
            strip_input_diacritics=True,
            normalization_options=normalize_options
        )

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        e = self.entries[idx]
        raw_text = e["raw"]
        
        # 1. Base Tensors (Preserved)
        chars = torch.tensor(e["char_ids"], dtype=torch.long)
        labels = torch.tensor(e["label_ids"], dtype=torch.long)
        
        # 2. Word IDs (New Logic: Aligned with Chars)
        word_ids = torch.zeros_like(chars)
        if self.word2idx:
            words = raw_text.split()
            w_ptr = 0
            for i, ch in enumerate(raw_text):
                if ch == ' ': 
                    w_ptr += 1
                elif w_ptr < len(words):
                    word_ids[i] = self.word2idx.get(words[w_ptr], 1) # 1 is UNK

        # 3. Sentence Features (New Logic)
        bow_vec = feature_mgr.transform_bow(raw_text) if cfg.use_bow else torch.empty(0)
        tfidf_vec = feature_mgr.transform_tfidf(raw_text) if cfg.use_tfidf else torch.empty(0)

        # Handle missing vectors
        if bow_vec is None: bow_vec = torch.empty(0)
        if tfidf_vec is None: tfidf_vec = torch.empty(0)

        return {
            "chars": chars,
            "labels": labels,
            "word_ids": word_ids,
            "bow": bow_vec,
            "tfidf": tfidf_vec,
            "raw": e["raw"],
            "label_strs": e["labels"]
        }