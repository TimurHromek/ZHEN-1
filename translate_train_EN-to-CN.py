import os
import math
import logging
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tqdm import tqdm

# --- Configuration ---
CONFIG = {
    "SRC_LANG": "en",
    "TGT_LANG": "zh",
    "TOKENIZER_FILE": "opus_en_zh_tokenizer.json",
    "MAX_SEQ_LEN": 128,
    "VOCAB_SIZE": 32000,
    "DIM": 256,
    "ENCODER_LAYERS": 4,
    "DECODER_LAYERS": 4,
    "N_HEADS": 8,
    "FF_DIM": 512,
    "DROPOUT": 0.1,
    "BATCH_SIZE": 64,
    "LEARNING_RATE": 5e-4,
    "NUM_EPOCHS": 5,
    "CHECKPOINT_DIR": "checkpoints_translation",
}

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Tokenizer Manager ---
class TokenizerManager:
    # ... (No changes needed in this class)
    def __init__(self, config):
        self.config = config
        self.tokenizer_path = Path(self.config["TOKENIZER_FILE"])
        self.special_tokens = ["<unk>", "<pad>", "<s>", "</s>"]
    def get_text_iterator(self):
        dataset = load_dataset(f"Helsinki-NLP/opus-100", f"{self.config['SRC_LANG']}-{self.config['TGT_LANG']}", split="train", streaming=True)
        for item in dataset: yield item['translation'][self.config['SRC_LANG']]; yield item['translation'][self.config['TGT_LANG']]
    def train_tokenizer(self):
        logging.info("Training a new tokenizer...")
        tokenizer = Tokenizer(BPE(unk_token="<unk>")); tokenizer.pre_tokenizer = Whitespace()
        trainer = BpeTrainer(vocab_size=self.config["VOCAB_SIZE"], special_tokens=self.special_tokens)
        tokenizer.train_from_iterator(self.get_text_iterator(), trainer=trainer)
        tokenizer.save(str(self.tokenizer_path)); logging.info(f"Tokenizer trained and saved to {self.tokenizer_path}")
        return tokenizer
    def get_tokenizer(self):
        if not self.tokenizer_path.exists(): return self.train_tokenizer()
        logging.info(f"Loading existing tokenizer from {self.tokenizer_path}")
        return Tokenizer.from_file(str(self.tokenizer_path))

# --- Dataset and Dataloader ---
class OpusDataset(Dataset):
    # ... (No changes needed in this class)
    def __init__(self, tokenizer, config, split="train"):
        self.tokenizer = tokenizer; self.config = config
        dataset = load_dataset(f"Helsinki-NLP/opus-100", f"{config['SRC_LANG']}-{config['TGT_LANG']}", split=split)
        self.pairs = [item['translation'] for item in dataset]
        self.src_lang, self.tgt_lang, self.max_len = config["SRC_LANG"], config["TGT_LANG"], config["MAX_SEQ_LEN"]
        self.bos_id, self.eos_id, self.pad_id = tokenizer.token_to_id("<s>"), tokenizer.token_to_id("</s>"), tokenizer.token_to_id("<pad>")
    def __len__(self): return len(self.pairs)
    def __getitem__(self, idx):
        pair = self.pairs[idx]
        src_text, tgt_text = pair[self.src_lang], pair[self.tgt_lang]
        src_tokens = [self.bos_id] + self.tokenizer.encode(src_text).ids + [self.eos_id]
        tgt_tokens = [self.bos_id] + self.tokenizer.encode(tgt_text).ids + [self.eos_id]
        return {"src": torch.tensor(src_tokens[:self.max_len], dtype=torch.long), "tgt": torch.tensor(tgt_tokens[:self.max_len], dtype=torch.long)}

class PadCollate:
    # ... (No changes needed in this class)
    def __init__(self, pad_id): self.pad_id = pad_id
    def __call__(self, batch):
        src_batch, tgt_batch = [item["src"] for item in batch], [item["tgt"] for item in batch]
        src_padded = pad_sequence(src_batch, batch_first=True, padding_value=self.pad_id)
        tgt_padded = pad_sequence(tgt_batch, batch_first=True, padding_value=self.pad_id)
        return {"src": src_padded, "tgt": tgt_padded}

# --- Model Architecture ---
class PositionalEncoding(nn.Module):

    def __init__(self, dim, dropout, max_len=5000):
        super().__init__(); self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1); div_term = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim))
        pe = torch.zeros(max_len, 1, dim); pe[:, 0, 0::2] = torch.sin(position * div_term); pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    def forward(self, x): x = x + self.pe[:x.size(0)]; return self.dropout(x)

class TranslationTransformer(nn.Module):

    def __init__(self, vocab_size, dim, n_heads, encoder_layers, decoder_layers, ff_dim, dropout, max_len):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim); self.pos_encoder = PositionalEncoding(dim, dropout, max_len)
        self.transformer = nn.Transformer(d_model=dim, nhead=n_heads, num_encoder_layers=encoder_layers, num_decoder_layers=decoder_layers, dim_feedforward=ff_dim, dropout=dropout, batch_first=True)
        self.generator = nn.Linear(dim, vocab_size)
    def _generate_mask(self, src, tgt, pad_id):
        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt.shape[1], device=tgt.device)
        src_padding_mask, tgt_padding_mask = (src == pad_id), (tgt == pad_id)
        return tgt_mask, src_padding_mask, tgt_padding_mask
    def forward(self, src, tgt, pad_id):
        src_emb = self.pos_encoder((self.embedding(src) * math.sqrt(CONFIG["DIM"])).permute(1, 0, 2)).permute(1, 0, 2)
        tgt_emb = self.pos_encoder((self.embedding(tgt) * math.sqrt(CONFIG["DIM"])).permute(1, 0, 2)).permute(1, 0, 2)
        tgt_mask, src_padding_mask, tgt_padding_mask = self._generate_mask(src, tgt, pad_id)
        output = self.transformer(src_emb, tgt_emb, tgt_mask=tgt_mask, src_key_padding_mask=src_padding_mask, tgt_key_padding_mask=tgt_padding_mask, memory_key_padding_mask=src_padding_mask)
        return self.generator(output)

# --- Trainer ---
class Trainer:
    def __init__(self, model, tokenizer, config):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=config["LEARNING_RATE"])
        self.pad_id = tokenizer.token_to_id("<pad>")
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.pad_id)
        self.scaler = torch.cuda.amp.GradScaler(enabled=(self.device.type == 'cuda'))
        self.checkpoint_dir = Path(config["CHECKPOINT_DIR"])
        self.checkpoint_dir.mkdir(exist_ok=True)

    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {self.current_epoch+1}/{self.config['NUM_EPOCHS']} Training")
        for batch in progress_bar:
            src, tgt = batch["src"].to(self.device), batch["tgt"].to(self.device)
            tgt_input, tgt_output = tgt[:, :-1], tgt[:, 1:]
            self.optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=self.device.type, enabled=(self.device.type == 'cuda')):
                logits = self.model(src, tgt_input, self.pad_id)
                loss = self.criterion(logits.view(-1, logits.size(-1)), tgt_output.reshape(-1))
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            total_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
        return total_loss / len(dataloader)

    # <<< NEW METHOD: For validation and testing >>>
    def evaluate(self, dataloader, description="Evaluating"):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            progress_bar = tqdm(dataloader, desc=description)
            for batch in progress_bar:
                src, tgt = batch["src"].to(self.device), batch["tgt"].to(self.device)
                tgt_input, tgt_output = tgt[:, :-1], tgt[:, 1:]
                logits = self.model(src, tgt_input, self.pad_id)
                loss = self.criterion(logits.view(-1, logits.size(-1)), tgt_output.reshape(-1))
                total_loss += loss.item()
                progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
        return total_loss / len(dataloader)

    def save_checkpoint(self, epoch, val_loss):
        filename = f"checkpoint_epoch_{epoch+1}_valloss_{val_loss:.4f}.pt"
        path = self.checkpoint_dir / filename
        torch.save({'epoch': epoch, 'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(), 'loss': val_loss}, path)
        logging.info(f"Checkpoint saved to {path}")


    def train(self, train_loader, val_loader):
        for epoch in range(self.config["NUM_EPOCHS"]):
            self.current_epoch = epoch
            logging.info(f"--- Starting Epoch {epoch + 1}/{self.config['NUM_EPOCHS']} ---")
            train_loss = self.train_epoch(train_loader)
            val_loss = self.evaluate(val_loader, description=f"Epoch {epoch+1}/{self.config['NUM_EPOCHS']} Validation")
            logging.info(f"Epoch {epoch+1} -> Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")
            self.save_checkpoint(epoch, val_loss)
            self.translate("This is a test to see how the model is learning.")

    def translate(self, src_sentence: str):
        self.model.eval()
        src_tokens = [self.tokenizer.token_to_id("<s>")] + self.tokenizer.encode(src_sentence).ids + [self.tokenizer.token_to_id("</s>")]
        src = torch.tensor(src_tokens, dtype=torch.long).unsqueeze(0).to(self.device)
        tgt_tokens = [self.tokenizer.token_to_id("<s>")]
        with torch.no_grad():
            for _ in range(self.config["MAX_SEQ_LEN"]):
                tgt_input = torch.tensor(tgt_tokens, dtype=torch.long).unsqueeze(0).to(self.device)
                logits = self.model(src, tgt_input, self.pad_id)
                next_token_id = logits[:, -1, :].argmax(dim=-1).item()
                tgt_tokens.append(next_token_id)
                if next_token_id == self.tokenizer.token_to_id("</s>"): break
        translated_text = self.tokenizer.decode(tgt_tokens, skip_special_tokens=True)
        logging.info(f"Source:      '{src_sentence}'")
        logging.info(f"Translated:  '{translated_text}'")

def main():
    # Implemented a cuda check to see if my drivers are turning schizo again or not.
    print("-" * 50)
    print("CUDA Health Check:")
    if torch.cuda.is_available():
        print(f"✅ CUDA is available.")
        print(f"   PyTorch Version: {torch.__version__}")
        print(f"   CUDA Version PyTorch was built with: {torch.version.cuda}")
        print(f"   Number of GPUs: {torch.cuda.device_count()}")
        print(f"   Current GPU Name: {torch.cuda.get_device_name(0)}")
    else:
        print(f"❌ CUDA is NOT available.")
        print(f"   PyTorch will run on CPU, which will be very slow.")
        print(f"   ACTION: Ensure you have installed PyTorch with CUDA support. See https://pytorch.org/get-started/locally/")
    print("-" * 50)

    tokenizer_manager = TokenizerManager(CONFIG)
    tokenizer = tokenizer_manager.get_tokenizer()
    CONFIG["VOCAB_SIZE"] = tokenizer.get_vocab_size()
    
    logging.info("Loading and preparing datasets...")
    train_dataset = OpusDataset(tokenizer, CONFIG, split="train")
    val_dataset = OpusDataset(tokenizer, CONFIG, split="validation")
    test_dataset = OpusDataset(tokenizer, CONFIG, split="test")
    logging.info(f"Dataset sizes -> Train: {len(train_dataset)}, Validation: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    pad_id = tokenizer.token_to_id("<pad>")
    collate_fn = PadCollate(pad_id)
    num_workers = 0 if os.name == 'nt' else os.cpu_count() // 2
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG["BATCH_SIZE"], shuffle=True, collate_fn=collate_fn, num_workers=num_workers, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["BATCH_SIZE"], shuffle=False, collate_fn=collate_fn, num_workers=num_workers, pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(test_dataset, batch_size=CONFIG["BATCH_SIZE"], shuffle=False, collate_fn=collate_fn, num_workers=num_workers, pin_memory=torch.cuda.is_available())

    model = TranslationTransformer(vocab_size=CONFIG["VOCAB_SIZE"], dim=CONFIG["DIM"], n_heads=CONFIG["N_HEADS"],
                                   encoder_layers=CONFIG["ENCODER_LAYERS"], decoder_layers=CONFIG["DECODER_LAYERS"],
                                   ff_dim=CONFIG["FF_DIM"], dropout=CONFIG["DROPOUT"], max_len=CONFIG["MAX_SEQ_LEN"])
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Model initialized. Total trainable parameters: {total_params:,}")

    trainer = Trainer(model, tokenizer, CONFIG)
    

    trainer.train(train_loader, val_loader)
    
    # NEW TESTS, NOT AS SHITTY AS BEFORE
    logging.info("\n--- Training Complete. Evaluating on Test Set... ---")
    test_loss = trainer.evaluate(test_loader, description="Final Test Evaluation")
    logging.info(f"Final Test Loss: {test_loss:.4f}")

    logging.info("\n--- Final Translation Examples ---")
    trainer.translate("The European Economic Area was created in 1994.")
    trainer.translate("What is your name?")
    trainer.translate("This technology is changing the world.")

if __name__ == "__main__":
    main()
