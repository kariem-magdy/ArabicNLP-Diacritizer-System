import torch
import os
import sys
from transformers import AutoTokenizer, AutoModelForTokenClassification
from ..config import cfg
from ..preprocess import load_json, extract_labels, normalize_text

class TransformerPredictor:
    def __init__(self):
        print(f"[INFO] Loading Transformer resources on {cfg.device}...")
        self.device = cfg.device
        
        # 1. Load Vocabs
        try:
            self.label2idx = load_json(os.path.join(cfg.processed_dir, "label2idx.json"))
            # Ensure keys are integers for decoding
            self.idx2label = {int(v): k for k, v in self.label2idx.items()}
        except FileNotFoundError:
            raise RuntimeError("Vocab files not found. Run build_vocab.py first.")

        # 2. Load Model & Tokenizer
        model_path = os.path.join(cfg.models_dir, "best_transformer")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"❌ Trained model not found at {model_path}. You must train the model first!")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForTokenClassification.from_pretrained(model_path).to(self.device)
        
        self.model.eval()
        print("[INFO] Transformer Model loaded successfully.")

    def predict(self, text):
        # 1. Preprocess
        # Clean text and remove existing diacritics
        text_clean, _ = extract_labels(text)
        
        # Apply normalization (Must match training!)
        norm_opts = {"normalize_hamza": True, "remove_tatweel": False, "lower_latin": True}
        text_clean = normalize_text(text_clean, norm_opts)
        
        if not text_clean: return ""

        # 2. Prepare Input (The Fix)
        # We send ONLY valid characters to the model to avoid space tokenization issues
        chars_only = [c for c in text_clean if c != ' ']
        
        if not chars_only: return ""

        inputs = self.tokenizer(
            chars_only, 
            is_split_into_words=True, 
            return_tensors="pt", 
            padding=True, 
            truncation=True
        ).to(self.device)

        # 3. Inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=2)

        # 4. Align Sub-tokens to Characters
        word_ids = inputs.word_ids()
        predicted_labels = []
        previous_word_idx = None
        
        for idx, word_idx in enumerate(word_ids):
            # Skip special tokens [CLS], [SEP]
            if word_idx is None:
                continue
            
            # Take the prediction of the FIRST sub-token for each character
            if word_idx != previous_word_idx:
                pred_id = predictions[0][idx].item()
                label = self.idx2label.get(pred_id, '_')
                predicted_labels.append(label)
                previous_word_idx = word_idx

        # 5. Reconstruct Sentence (Merge Spaces + Diacritics)
        out_str = ""
        pred_ptr = 0
        
        for char in text_clean:
            if char == ' ':
                out_str += ' '
                # Do NOT increment pred_ptr, as spaces were not sent to the model
            else:
                if pred_ptr < len(predicted_labels):
                    diacritic = predicted_labels[pred_ptr]
                    out_str += char + (diacritic if diacritic != '_' else '')
                    pred_ptr += 1
                else:
                    # Fallback if something went wrong (rare)
                    out_str += char
            
        return out_str

if __name__ == "__main__":
    try:
        predictor = TransformerPredictor()
        text = "ذهب علي الي الشاطئ"
        print(f"\nInput:  {text}")
        print(f"Output: {predictor.predict(text)}")
    except Exception as e:
        print(f"Error: {e}")