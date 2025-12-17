import torch
import os
import csv
import pickle
from tqdm import tqdm
from src.infer.infer import DiacriticPredictor
from src.features import feature_mgr
from src.config import cfg

# --- 1. Load TA's Resources ---
# We load the allowed characters to filter out numbers, brackets, and spaces exactly like the Golden File.
try:
    with open("arabic_letters.pickle", "rb") as f:
        ALLOWED_CHARS = set(pickle.load(f))
    print(f"[INFO] Loaded {len(ALLOWED_CHARS)} allowed characters from arabic_letters.pickle")
except FileNotFoundError:
    print("❌ Error: 'arabic_letters.pickle' not found. Please upload it to the project root.")
    exit(1)

# --- 2. Define the Competition Mapping ---
COMPETITION_MAP = {
    'َ': 0, 
    'ً': 1, 
    'ُ': 2, 
    'ٌ': 3, 
    'ِ': 4, 
    'ٍ': 5, 
    'ْ': 6, 
    'ّ': 7, 
    'َّ': 8, 
    'ًّ': 9, 
    'ُّ': 10, 
    'ٌّ': 11, 
    'ِّ': 12, 
    'ٍّ': 13, 
    '': 14, 
    '_': 14 
}

def generate_submission(input_file, output_file):
    # Initialize Predictor (loads model, vocabs, and feature_mgr)
    predictor = DiacriticPredictor()
    
    print(f"Reading from: {input_file}")
    print(f"Writing to:   {output_file}")
    
    results = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    for line in tqdm(lines, desc="Predicting"):
        line = line.strip()
        if not line: continue
        
        # A. Preprocess (Normalize)
        # We use the same normalization as training to get the correct model input
        from src.preprocess import extract_labels, normalize_text
        text_clean, _ = extract_labels(line)
        text_clean = normalize_text(text_clean, {
            "normalize_hamza": True, 
            "remove_tatweel": True, 
            "lower_latin": True, 
            "remove_punctuation": True
        })
        
        if not text_clean: continue

        # B. Prepare ALL Features (Chars + Words + BoW + TFIDF)
        
        # 1. Chars
        chars = torch.tensor([predictor.char2idx.get(c, 1) for c in text_clean], dtype=torch.long).unsqueeze(0).to(predictor.device)
        
        # 2. Words
        word_ids = None
        if predictor.word2idx:
            w_ids = []
            words = text_clean.split()
            ptr = 0
            for c in text_clean:
                if c == ' ': 
                    w_ids.append(0)
                    ptr += 1
                elif ptr < len(words):
                    w_ids.append(predictor.word2idx.get(words[ptr], 1))
                else:
                    w_ids.append(0)
            word_ids = torch.tensor(w_ids, dtype=torch.long).unsqueeze(0).to(predictor.device)

        # 3. Sentence Features
        bow = feature_mgr.transform_bow(text_clean).unsqueeze(0).to(predictor.device) if cfg.use_bow else None
        tfidf = feature_mgr.transform_tfidf(text_clean).unsqueeze(0).to(predictor.device) if cfg.use_tfidf else None

        mask = torch.ones_like(chars, dtype=torch.bool)
        
        # C. Forward Pass
        with torch.no_grad():
            pred_indices = predictor.model(chars, word_ids, bow, tfidf, mask=mask)[0]
            
        # D. Translate to Competition IDs (With Strict Filtering)
        line_ids = []
        for char, pid in zip(text_clean, pred_indices):
            # --- CRITICAL FIX ---
            # If the character is NOT in the TA's allowed list (e.g. Spaces, Numbers, Brackets)
            # We SKIP it completely. This aligns your IDs with the Golden File.
            if char not in ALLOWED_CHARS:
                continue

            # Get Label
            label_str = predictor.idx2label.get(int(pid), '_')
            comp_id = COMPETITION_MAP.get(label_str, 14) 
            line_ids.append(comp_id)
            
        results.extend(line_ids)

    # Save to CSV
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["ID", "Label"])
        for i, val in enumerate(results):
            writer.writerow([i, val])
            
    print("✅ Submission file generated successfully!")

if __name__ == "__main__":
    TEST_FILE = "data/test.txt"     
    OUTPUT_CSV = "submission_3ntr_1.csv"   
    
    if os.path.exists(TEST_FILE):
        generate_submission(TEST_FILE, OUTPUT_CSV)
    else:
        print(f"❌ Error: Could not find {TEST_FILE}")