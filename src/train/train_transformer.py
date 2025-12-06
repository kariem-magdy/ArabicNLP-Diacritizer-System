import os
import torch
import logging
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForTokenClassification, 
    TrainingArguments, 
    Trainer,
    DataCollatorForTokenClassification
)
from ..config import cfg
from ..preprocess import load_json, parse_file_to_entries

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TrainTransformer")

def run():
    os.makedirs(cfg.models_dir, exist_ok=True)
    
    # 1. Load Mappings
    label2idx_path = os.path.join(cfg.outputs_dir, "processed/label2idx.json")
    char2idx_path = os.path.join(cfg.outputs_dir, "processed/char2idx.json")
    
    if not os.path.exists(label2idx_path):
        raise FileNotFoundError("Run preprocessing first to generate label2idx.json")
        
    label2idx = load_json(label2idx_path)
    char2idx = load_json(char2idx_path) 
    
    idx2label = {int(v): k for k, v in label2idx.items()}
    label_list = [idx2label[i] for i in range(len(idx2label))]
    
    logger.info(f"Loaded {len(label2idx)} labels.")

    # 2. Prepare Data
    # Use same normalization as BiLSTM to ensure consistency
    norm_opts = {
        "normalize_hamza": True, 
        "remove_tatweel": False, 
        "lower_latin": True, 
        "remove_punctuation": True
    }

    logger.info("Parsing Train Data with Normalization...")
    train_entries = parse_file_to_entries(cfg.train_file, char2idx, label2idx, normalization_options=norm_opts)
    
    logger.info("Parsing Val Data with Normalization...")
    val_entries = parse_file_to_entries(cfg.val_file, char2idx, label2idx, normalization_options=norm_opts)

    def format_for_hf(entries):
        return {
            "tokens": [list(e['raw']) for e in entries], 
            "ner_tags": [e['label_ids'] for e in entries]
        }

    train_ds = Dataset.from_dict(format_for_hf(train_entries))
    val_ds = Dataset.from_dict(format_for_hf(val_entries))

    # 3. Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.transformer_model_name, use_fast=True)

    # 4. Alignment (Crucial for char-level tasks)
    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(
            examples["tokens"], 
            truncation=True, 
            is_split_into_words=True, 
            max_length=512,
            padding="max_length"
        )

        labels = []
        for i, label in enumerate(examples["ner_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100) # Special tokens/Padding -> Masked
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx]) # First sub-token -> Label
                else:
                    label_ids.append(-100) # Subsequent sub-tokens -> Masked
                previous_word_idx = word_idx
            labels.append(label_ids)

        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    train_tokenized = train_ds.map(tokenize_and_align_labels, batched=True)
    val_tokenized = val_ds.map(tokenize_and_align_labels, batched=True)

    # 5. Model
    model = AutoModelForTokenClassification.from_pretrained(
        cfg.transformer_model_name, 
        num_labels=len(label_list),
        id2label=idx2label,
        label2id=label2idx
    )

    # 6. Metrics (DER with Masking)
    def compute_metrics(p):
        predictions, labels = p
        predictions = torch.tensor(predictions).argmax(dim=2)
        labels = torch.tensor(labels)

        # Filter out -100. This IS the mask.
        # It removes padding, special tokens, and sub-token continuations.
        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        flat_preds = [item for sublist in true_predictions for item in sublist]
        flat_labels = [item for sublist in true_labels for item in sublist]
        
        correct = sum(1 for p, l in zip(flat_preds, flat_labels) if p == l)
        total = len(flat_labels)
        accuracy = correct / total if total > 0 else 0
        der = (1 - accuracy) * 100
        
        return {"accuracy": accuracy, "der": der}

    # 7. Trainer
    training_args = TrainingArguments(
        output_dir=cfg.models_dir,
        # FIX: Changed evaluation_strategy -> eval_strategy
        eval_strategy="epoch", 
        save_strategy="epoch",
        learning_rate=cfg.transformer_lr,
        per_device_train_batch_size=cfg.transformer_batch_size,
        per_device_eval_batch_size=cfg.transformer_batch_size,
        num_train_epochs=cfg.transformer_epochs,
        weight_decay=0.01,
        save_total_limit=1,
        logging_dir=cfg.logs_dir,
        load_best_model_at_end=True,
        metric_for_best_model="der",
        greater_is_better=False,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
        tokenizer=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_metrics=compute_metrics
    )

    logger.info("Starting Training...")
    trainer.train()
    
    trainer.save_model(os.path.join(cfg.models_dir, "best_transformer"))
    logger.info(f"Model saved to {cfg.models_dir}")

if __name__ == "__main__":
    run()