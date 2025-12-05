# src/train/train_bilstm.py
import os
import torch
import logging
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from ..config import cfg
from ..preprocess import load_json
from ..data.dataset import DiacritizationDataset
from ..data.collate import collate_fn
from ..models.bilstm_crf import BiLSTMCRF
from ..eval.metrics import DERCalculator
from ..utils.checkpoints import save_checkpoint

# Configure Logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TrainBiLSTM")

def train():
    os.makedirs(cfg.models_dir, exist_ok=True)
    
    # 1. Load Vocabs (Must be pre-built using preprocess.py)
    # Ensure you ran preprocess.py or main.py first!
    char2idx_path = os.path.join(cfg.outputs_dir, "processed/char2idx.json")
    label2idx_path = os.path.join(cfg.outputs_dir, "processed/label2idx.json")
    
    if not os.path.exists(char2idx_path) or not os.path.exists(label2idx_path):
        raise FileNotFoundError(f"Vocab files not found at {cfg.outputs_dir}. Run preprocessing first.")
        
    char2idx = load_json(char2idx_path)
    label2idx = load_json(label2idx_path)
    
    logger.info(f"Loaded Vocabs: {len(char2idx)} chars, {len(label2idx)} labels")

    # 2. Create Datasets
    # Pass dicts directly; logic is delegated to preprocess.parse_file_to_entries
    train_ds = DiacritizationDataset(cfg.train_file, char2idx, label2idx)
    dev_ds = DiacritizationDataset(cfg.dev_file, char2idx, label2idx)

    # 3. DataLoaders
    # Using the updated collate_fn from src.data.collate
    train_loader = DataLoader(
        train_ds, 
        batch_size=cfg.batch_size, 
        shuffle=True, 
        collate_fn=collate_fn, 
        num_workers=2
    )
    dev_loader = DataLoader(
        dev_ds, 
        batch_size=cfg.batch_size, 
        shuffle=False, 
        collate_fn=collate_fn, 
        num_workers=2
    )

    # 4. Model Initialization
    # pad_idx is 0 in our preprocess.py
    model = BiLSTMCRF(
        vocab_size=len(char2idx), 
        char_emb_dim=cfg.char_emb_dim, 
        lstm_hidden=cfg.lstm_hidden, 
        num_labels=len(label2idx), 
        num_layers=cfg.lstm_layers, 
        dropout=cfg.bilstm_dropout, 
        pad_idx=0
    )
    
    device = torch.device(cfg.device)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    
    # 5. Training Loop
    best_der = 100.0
    der_calc = DERCalculator() 
    # Map indices back to strings for DER calculation
    idx2label = {v: k for k, v in label2idx.items()}

    logger.info("Starting Training...")
    
    for epoch in range(cfg.epochs):
        model.train()
        total_loss = 0.0
        
        # Train Step
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.epochs}"):
            chars = batch['chars'].to(device)
            labels = batch['labels'].to(device)
            mask = batch['mask'].to(device) # Boolean mask from collate_fn
            
            optimizer.zero_grad()
            # Forward returns loss because labels are provided
            loss = model(chars, labels=labels, mask=mask)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        
        # Eval Step
        model.eval()
        all_preds = [] # List of list of strings
        all_refs = []  # List of list of strings
        
        with torch.no_grad():
            for batch in dev_loader:
                chars = batch['chars'].to(device)
                mask = batch['mask'].to(device)
                
                # Decode: returns list of list of label indices
                batch_preds_ids = model(chars, mask=mask)
                
                # Retrieve raw labels (strings) from dataset for accurate comparison
                # Or decode the label_ids using idx2label
                batch_ref_ids = batch['labels'].cpu().tolist()
                
                # Align lengths using the mask or the raw prediction output
                for i, pred_ids in enumerate(batch_preds_ids):
                    # Filter padding from reference using mask
                    valid_len = len(pred_ids) 
                    ref_ids = batch_ref_ids[i][:valid_len]
                    
                    pred_labels = [idx2label.get(pid, '_') for pid in pred_ids]
                    ref_labels = [idx2label.get(rid, '_') for rid in ref_ids]
                    
                    all_preds.append(pred_labels)
                    all_refs.append(ref_labels)

        # Calculate DER (Diacritic Error Rate)
        # Assuming DERCalculator takes lists of lists of label strings
        # Note: We must pass raw text if DERCalculator expects it, but usually comparing label lists is enough.
        # If your DERCalculator requires full sentences, you can reconstruct them using batch['raws']
        
        # Simplified DER calculation here for safety if metrics.py varies:
        flat_preds = [l for seq in all_preds for l in seq]
        flat_refs = [l for seq in all_refs for l in seq]
        correct = sum(1 for p, r in zip(flat_preds, flat_refs) if p == r)
        total = len(flat_refs)
        current_der = 100.0 * (1.0 - (correct / total)) if total > 0 else 0.0

        logger.info(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Dev DER={current_der:.2f}%")

        if current_der < best_der:
            best_der = current_der
            save_path = os.path.join(cfg.models_dir, 'best_bilstm.pt')
            save_checkpoint({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'char2idx': char2idx,
                'label2idx': label2idx,
                'best_der': best_der
            }, save_path)
            logger.info(f"New best model saved to {save_path}")

if __name__ == "__main__":
    train()