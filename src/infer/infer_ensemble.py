# src/infer/infer_ensemble.py
import torch
import pickle
import json
import os
import numpy as np

from ..models.bilstm_crf import BiLSTMCRF
from ..models.transformer_finetune import TransformerTokenClassifier
from ..config import cfg
from ..features import FeatureGenerator

class EnsemblePredictor:
    def __init__(self, bilstm_path, transformer_path, artifacts_path):
        self.device = torch.device(cfg.device)
        print(f"Loading Ensemble on {self.device}...")
        
        # 1. Load Vocabs
        with open(os.path.join(artifacts_path, 'char2idx.json'), 'r') as f:
            self.char2idx = json.load(f)
        with open(os.path.join(artifacts_path, 'idx2label.json'), 'r') as f:
            self.idx2label = json.load(f)
        with open(os.path.join(artifacts_path, 'label2idx.json'), 'r') as f:
            self.label2idx = json.load(f)
        
        # 2. Load Vectorizers
        bow_vec = None
        tfidf_vec = None
        if cfg.use_bow and os.path.exists(os.path.join(artifacts_path, 'bow.pkl')):
            with open(os.path.join(artifacts_path, 'bow.pkl'), 'rb') as f:
                bow_vec = pickle.load(f)
        if cfg.use_tfidf and os.path.exists(os.path.join(artifacts_path, 'tfidf.pkl')):
            with open(os.path.join(artifacts_path, 'tfidf.pkl'), 'rb') as f:
                tfidf_vec = pickle.load(f)
        
        # 3. Load Word Vocab
        word2idx = None
        word_vocab_size = None
        if cfg.use_word_emb and os.path.exists(os.path.join(artifacts_path, 'word2idx.json')):
            with open(os.path.join(artifacts_path, 'word2idx.json'), 'r') as f:
                word2idx = json.load(f)
                word_vocab_size = len(word2idx)

        # 4. Load BiLSTM
        self.bilstm_model = BiLSTMCRF(
            vocab_size=len(self.char2idx), 
            char_emb_dim=cfg.char_emb_dim, 
            lstm_hidden=cfg.lstm_hidden, 
            num_labels=len(self.label2idx),
            word_vocab_size=word_vocab_size,
            fasttext_matrix=None, 
            num_layers=cfg.lstm_layers,
            dropout=cfg.bilstm_dropout
        ).to(self.device)
        
        if os.path.exists(bilstm_path):
            checkpoint = torch.load(bilstm_path, map_location=self.device)
            state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
            self.bilstm_model.load_state_dict(state_dict)
            self.bilstm_model.eval()
        else:
            print(f"[WARN] BiLSTM model not found at {bilstm_path}")
        
        # 5. Load Transformer
        self.transformer_classifier = TransformerTokenClassifier(
            model_name=cfg.transformer_model_name,
            label_list=list(self.label2idx.keys()),
            device=self.device
        )
        if os.path.exists(transformer_path):
            self.transformer_classifier.load(transformer_path)
        else:
            print(f"[WARN] Transformer model not found at {transformer_path}")
        
        # 6. Setup Generator
        self.feature_generator = FeatureGenerator(
            char2idx=self.char2idx, 
            word2idx=word2idx, 
            bow_vectorizer=bow_vec, 
            tfidf_vectorizer=tfidf_vec,
            device=self.device
        )

    def predict(self, sentence: str, ensemble_weight: float = 0.5):
        """
        Runs both models.
        ensemble_weight: < 0.5 favors BiLSTM, > 0.5 favors Transformer.
        Returns: Words, BiLSTM_Tags, Transformer_Tags, Final_Tags (Merged)
        """
        sentence_words = sentence.strip().split()
        if not sentence_words: return [], [], [], []

        # --- A. BiLSTM Inference (Char Level) ---
        bilstm_char_tags = []
        try:
            with torch.no_grad():
                inputs = self.feature_generator.generate_inputs_from_words([sentence_words])
                indices_list = self.bilstm_model(*inputs)
                bilstm_preds = indices_list[0].cpu().tolist() # List of char labels
                bilstm_char_tags = [self.idx2label[str(idx)] for idx in bilstm_preds]
        except Exception as e:
            print(f"BiLSTM Error: {e}")

        # Group BiLSTM Char Tags by Word
        # (Since we flattened words to chars earlier, we reconstruct grouping)
        bilstm_word_tags = []
        char_ptr = 0
        for w in sentence_words:
            length = len(w)
            # Safely slice tags for this word
            if char_ptr + length <= len(bilstm_char_tags):
                tags_for_word = bilstm_char_tags[char_ptr : char_ptr + length]
                bilstm_word_tags.append(tags_for_word)
            else:
                bilstm_word_tags.append(["O"] * length)
            char_ptr += length

        # --- B. Transformer Inference (Subword Level) ---
        transformer_word_tags = [] 
        # Initialize with None to match length
        transformer_word_tags = [["O"] * len(w) for w in sentence_words]
        
        try:
            preds_indices, enc = self.transformer_classifier.predict_on_sentences([sentence])
            pred_indices = preds_indices[0] # First batch
            word_ids = enc.word_ids(batch_index=0)
            
            # Map subword labels to words. 
            # Strategy: If a word is split into subwords, we take the label of the subwords 
            # and try to assign them to the word's characters.
            # Simplified: We assign the label of the *first* subword token to *all* chars in the word
            # (or use it as a 'Word Level' tag if that's how your model was trained).
            
            curr_word_idx = None
            for idx, w_id in enumerate(word_ids):
                if w_id is None: continue # Special tokens
                
                # If this is a new word we haven't processed yet (or first part of it)
                # We extract the label
                tag_idx = pred_indices[idx]
                tag_str = self.idx2label[str(tag_idx)]
                
                # Check bounds
                if w_id < len(transformer_word_tags):
                    # We simply overwrite the tags for this word with the Transformer's prediction
                    # This assumes Transformer predicts a 'Whole Word' or 'Stem' tag.
                    # If Transformer is fine-grained, this logic needs 'offset_mapping', 
                    # but 'predict_on_sentences' doesn't return offsets by default.
                    transformer_word_tags[w_id] = [tag_str] * len(sentence_words[w_id])
                    
        except Exception as e:
            print(f"Transformer Error: {e}")

        # --- C. Ensemble / Merge ---
        final_tags_flat = []
        
        # We iterate word by word
        for i in range(len(sentence_words)):
            b_tags = bilstm_word_tags[i]      # List of char tags
            t_tags = transformer_word_tags[i] # List of char tags (derived from subwords)
            
            merged_word_tags = []
            
            # Iterate char by char within the word
            for j in range(len(b_tags)):
                b = b_tags[j]
                t = t_tags[j] if j < len(t_tags) else "O"
                
                if b == t:
                    merged_word_tags.append(b)
                else:
                    # Voting
                    merged_word_tags.append(t if ensemble_weight > 0.5 else b)
            
            final_tags_flat.extend(merged_word_tags)

        # For the App, we want to return Word-Level summary if possible, 
        # but Diacritization is Char-Level. 
        # So we return the List of Words, and the *Detailed Char Tags* grouped by word 
        # so the App can render them nicely.
        
        # Flatten for simple display if needed, or keep grouped.
        # Let's return simple lists of 'Representative Tags' (e.g. Last Char) for the table,
        # but the full tag lists are used internally.
        
        # To strictly match "infer it as bilstm infer", we probably want the list of 
        # tags for the *last character* of each word (Case Ending) or similar?
        # Or just return the flat list of char tags? 
        # I will return the "Representative Tag" (last char) for the table summary.
        
        b_summary = [t[-1] if t else "O" for t in bilstm_word_tags]
        t_summary = [t[-1] if t else "O" for t in transformer_word_tags]
        f_summary = []
        
        # Re-calculate final summary based on flattened logic
        ptr = 0
        for w in sentence_words:
            # Get the tags assigned to the last char of this word from final_tags_flat
            end_idx = ptr + len(w)
            last_tag = final_tags_flat[end_idx - 1] if end_idx <= len(final_tags_flat) else "O"
            f_summary.append(last_tag)
            ptr += len(w)
            
        return sentence_words, b_summary, t_summary, f_summary