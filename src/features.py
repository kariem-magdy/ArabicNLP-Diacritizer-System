# src/features.py
import os
import pickle
import numpy as np
import torch
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from .config import cfg

class FeatureManager:
    def __init__(self):
        self.bow_vectorizer = None
        self.tfidf_vectorizer = None
        
    def fit(self, texts):
        print("[INFO] Fitting BoW and TF-IDF Vectorizers...")
        if cfg.use_bow:
            self.bow_vectorizer = CountVectorizer(max_features=cfg.bow_vocab_size, binary=False)
            self.bow_vectorizer.fit(texts)
        if cfg.use_tfidf:
            self.tfidf_vectorizer = TfidfVectorizer(max_features=cfg.tfidf_vocab_size)
            self.tfidf_vectorizer.fit(texts)
        self.save()

    def transform_bow(self, text):
        if not self.bow_vectorizer: return torch.zeros(0)
        try:
            vec = self.bow_vectorizer.transform([text]).toarray()[0]
            return torch.tensor(vec, dtype=torch.float)
        except: return torch.zeros(0)

    def transform_tfidf(self, text):
        if not self.tfidf_vectorizer: return torch.zeros(0)
        try:
            vec = self.tfidf_vectorizer.transform([text]).toarray()[0]
            return torch.tensor(vec, dtype=torch.float)
        except: return torch.zeros(0)

    def save(self):
        os.makedirs(cfg.processed_dir, exist_ok=True)
        if self.bow_vectorizer:
            with open(os.path.join(cfg.processed_dir, 'bow.pkl'), 'wb') as f: pickle.dump(self.bow_vectorizer, f)
        if self.tfidf_vectorizer:
            with open(os.path.join(cfg.processed_dir, 'tfidf.pkl'), 'wb') as f: pickle.dump(self.tfidf_vectorizer, f)

    def load(self):
        bow_path = os.path.join(cfg.processed_dir, 'bow.pkl')
        tfidf_path = os.path.join(cfg.processed_dir, 'tfidf.pkl')
        try:
            if cfg.use_bow and os.path.exists(bow_path):
                with open(bow_path, 'rb') as f: self.bow_vectorizer = pickle.load(f)
            if cfg.use_tfidf and os.path.exists(tfidf_path):
                with open(tfidf_path, 'rb') as f: self.tfidf_vectorizer = pickle.load(f)
        except Exception as e:
            print(f"[WARN] Failed to load feature pickles: {e}")

    def load_fasttext_matrix(self, word2idx):
        if not os.path.exists(cfg.fasttext_path):
            print(f"[WARN] FastText file not found at {cfg.fasttext_path}. Skipping.")
            return None
        
        print("[INFO] Loading FastText...")
        vocab_size = len(word2idx)
        matrix = np.zeros((vocab_size, cfg.fasttext_dim))
        found = 0
        
        with open(cfg.fasttext_path, 'r', encoding='utf-8', errors='ignore') as f:
            for i, line in enumerate(f):
                if i == 0 and len(line.split()) < 10: continue
                parts = line.rstrip().split(' ')
                word = parts[0]
                if word in word2idx:
                    try:
                        matrix[word2idx[word]] = np.array(parts[1:], dtype=float)
                        found += 1
                    except: pass
        print(f"[INFO] FastText found {found} words.")
        return torch.tensor(matrix, dtype=torch.float)

feature_mgr = FeatureManager()

# --- NEW CLASS: FEATURE GENERATOR FOR INFERENCE ---
class FeatureGenerator:
    """
    Prepares tensors for BiLSTM inference by converting raw words 
    into Character IDs and Word IDs (aligned per character).
    """
    def __init__(self, char2idx, word2idx=None, bow_vectorizer=None, tfidf_vectorizer=None, device='cpu'):
        self.char2idx = char2idx
        self.word2idx = word2idx
        self.bow_vectorizer = bow_vectorizer
        self.tfidf_vectorizer = tfidf_vectorizer
        self.device = device
        self.pad_idx = 0

    def _get_word_ids(self, sentences_words):
        # BiLSTM operates on Character Sequences. 
        # We must repeat the Word ID for every character in that word.
        if not self.word2idx: return None
        
        word_sequences = []
        unk_idx = self.word2idx.get('<unk>', 1) 
        
        for words in sentences_words:
            seq_ids = []
            for w in words:
                w_id = self.word2idx.get(w, unk_idx)
                # CRITICAL: Repeat w_id for len(w) times to match char sequence
                seq_ids.extend([w_id] * len(w))
            word_sequences.append(seq_ids)
        
        # Padding
        max_len = max(len(s) for s in word_sequences)
        padded_sequences = [s + [self.pad_idx] * (max_len - len(s)) for s in word_sequences]
        return torch.tensor(padded_sequences, dtype=torch.long, device=self.device)

    def _get_char_ids(self, sentences_words):
        char_sequences = []
        unk_idx = self.char2idx.get('<unk>', 1) 

        for words in sentences_words:
            # Flatten words into characters
            char_ids = [self.char2idx.get(c, unk_idx) for word in words for c in word]
            char_sequences.append(char_ids)

        # Padding
        max_len = max(len(s) for s in char_sequences)
        padded_sequences = [s + [self.pad_idx] * (max_len - len(s)) for s in char_sequences]
        
        # Mask
        mask = [[1] * len(s) + [0] * (max_len - len(s)) for s in char_sequences]
        
        return (torch.tensor(padded_sequences, dtype=torch.long, device=self.device), 
                torch.tensor(mask, dtype=torch.bool, device=self.device))

    def _get_bow_features(self, sentences):
        if not self.bow_vectorizer: return None
        try:
            bow_matrix = self.bow_vectorizer.transform(sentences).toarray()
            return torch.tensor(bow_matrix, dtype=torch.float, device=self.device)
        except: return None

    def _get_tfidf_features(self, sentences):
        if not self.tfidf_vectorizer: return None
        try:
            tfidf_matrix = self.tfidf_vectorizer.transform(sentences).toarray()
            return torch.tensor(tfidf_matrix, dtype=torch.float, device=self.device)
        except: return None

    def generate_inputs_from_words(self, sentences_words):
        # sentences_words: List[List[str]] (batch of tokenized sentences)
        sentences = [" ".join(words) for words in sentences_words]
        
        chars_tensor, mask = self._get_char_ids(sentences_words)
        word_ids_tensor = self._get_word_ids(sentences_words)
        bow_tensor = self._get_bow_features(sentences)
        tfidf_tensor = self._get_tfidf_features(sentences)
        
        return (chars_tensor, word_ids_tensor, bow_tensor, tfidf_tensor, mask)