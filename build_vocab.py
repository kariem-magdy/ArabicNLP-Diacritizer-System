# build_vocab.py
import os
from src.config import cfg
from src.preprocess import clean_line, extract_labels, build_char_vocab, build_label_map, save_json, normalize_base_and_labels
from src.features import feature_mgr

# Options matching Dataset default
NORM_OPTS = {"normalize_hamza": True, "remove_tatweel": True, "lower_latin": True, "remove_punctuation": False}

def build_word_vocab(texts):
    print("[INFO] Building Word Vocabulary...")
    words = set()
    for t in texts:
        for w in t.split():
            words.add(w)
    word2idx = {"<PAD>": 0, "<UNK>": 1}
    for i, w in enumerate(sorted(list(words)), 2): word2idx[w] = i
    return word2idx

def build_vocabs():
    os.makedirs(cfg.processed_dir, exist_ok=True)
    print(f"[INFO] Reading {cfg.train_file}...")
    
    all_raw_normalized = []
    all_chars = []
    all_labels = []
    
    with open(cfg.train_file, "r", encoding="utf-8") as f:
        for line in f:
            line = clean_line(line.rstrip("\n"))
            if not line: continue
            
            base, labels = extract_labels(line)
            
            # IMPORTANT: Normalize base AND labels in sync using the new function
            base, labels = normalize_base_and_labels(base, labels, NORM_OPTS)
            
            if not base: continue
            
            all_chars.append(base)
            all_labels.append(labels)
            all_raw_normalized.append(base)
            
    # 1. Char Vocabs
    char2idx, idx2char = build_char_vocab(all_chars, min_freq=cfg.char_vocab_min_freq)
    label2idx, idx2label = build_label_map(all_labels)
    
    save_json(char2idx, os.path.join(cfg.processed_dir, "char2idx.json"))
    save_json(idx2char, os.path.join(cfg.processed_dir, "idx2char.json"))
    save_json(label2idx, os.path.join(cfg.processed_dir, "label2idx.json"))
    save_json(idx2label, os.path.join(cfg.processed_dir, "idx2label.json"))
    
    # 2. Fit Features on Normalized Text
    feature_mgr.fit(all_raw_normalized)
    
    # 3. Word Vocab
    if cfg.use_word_emb or cfg.use_fasttext:
        word2idx = build_word_vocab(all_raw_normalized)
        save_json(word2idx, os.path.join(cfg.processed_dir, "word2idx.json"))

    print("[SUCCESS] Build Complete.")

if __name__ == "__main__":
    build_vocabs()