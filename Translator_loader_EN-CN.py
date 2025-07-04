import torch
import torch.nn as nn
from pathlib import Path
import math
import logging
import re

# --- Setup ---
# Configure logging to be minimal for inference
logging.basicConfig(level=logging.INFO, format='%(message)s')

# --- Configuration (Must match the training script) ---
CONFIG = {
    "SRC_LANG": "en",
    "TGT_LANG": "zh",
    "TOKENIZER_FILE": "opus_en_zh_tokenizer.json",
    "MAX_SEQ_LEN": 128,
    "DIM": 256,
    "ENCODER_LAYERS": 4,
    "DECODER_LAYERS": 4,
    "N_HEADS": 8,
    "FF_DIM": 512,
    "DROPOUT": 0.1,
    "CHECKPOINT_DIR": "checkpoints_translation",
}


class PositionalEncoding(nn.Module):
    def __init__(self, dim, dropout, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim))
        pe = torch.zeros(max_len, 1, dim)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class TranslationTransformer(nn.Module):
    def __init__(self, vocab_size, dim, n_heads, encoder_layers, decoder_layers, ff_dim, dropout, max_len):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim)
        self.pos_encoder = PositionalEncoding(dim, dropout, max_len)
        self.transformer = nn.Transformer(
            d_model=dim, nhead=n_heads, num_encoder_layers=encoder_layers,
            num_decoder_layers=decoder_layers, dim_feedforward=ff_dim,
            dropout=dropout, batch_first=True
        )
        self.generator = nn.Linear(dim, vocab_size)

    def _generate_mask(self, src, tgt, pad_id):
        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt.shape[1], device=tgt.device)
        src_padding_mask = (src == pad_id)
        tgt_padding_mask = (tgt == pad_id)
        return tgt_mask, src_padding_mask, tgt_padding_mask

    def forward(self, src, tgt, pad_id):
        src_emb = self.pos_encoder((self.embedding(src) * math.sqrt(CONFIG["DIM"])).permute(1, 0, 2)).permute(1, 0, 2)
        tgt_emb = self.pos_encoder((self.embedding(tgt) * math.sqrt(CONFIG["DIM"])).permute(1, 0, 2)).permute(1, 0, 2)
        tgt_mask, src_padding_mask, tgt_padding_mask = self._generate_mask(src, tgt, pad_id)
        output = self.transformer(
            src_emb, tgt_emb, tgt_mask=tgt_mask, src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask, memory_key_padding_mask=src_padding_mask
        )
        return self.generator(output)

# We need to import the Tokenizer class to load the tokenizer file
from tokenizers import Tokenizer

class Translator:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {self.device}")

        # Load the trained tokenizer
        tokenizer_path = Path(self.config["TOKENIZER_FILE"])
        if not tokenizer_path.exists():
            raise FileNotFoundError(f"Tokenizer file not found at {tokenizer_path}. Please run the training script first.")
        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        
        # Get special token IDs
        self.bos_id = self.tokenizer.token_to_id("<s>")
        self.eos_id = self.tokenizer.token_to_id("</s>")
        self.pad_id = self.tokenizer.token_to_id("<pad>")

        # Initialize the model structure
        self.model = TranslationTransformer(
            vocab_size=self.tokenizer.get_vocab_size(),
            dim=self.config["DIM"], n_heads=self.config["N_HEADS"],
            encoder_layers=self.config["ENCODER_LAYERS"], decoder_layers=self.config["DECODER_LAYERS"],
            ff_dim=self.config["FF_DIM"], dropout=self.config["DROPOUT"], max_len=self.config["MAX_SEQ_LEN"]
        )
        self.model.to(self.device)

    def load_best_checkpoint(self):
        """Finds and loads the checkpoint with the lowest validation loss."""
        checkpoint_dir = Path(self.config["CHECKPOINT_DIR"])
        if not checkpoint_dir.exists():
            raise FileNotFoundError(f"Checkpoint directory not found at {checkpoint_dir}.")

        best_loss = float('inf')
        best_checkpoint_path = None
        
        for chk_path in checkpoint_dir.glob("*.pt"):
            # Use regex to find the validation loss in the filename
            match = re.search(r'valloss_([\d.]+)\.pt', chk_path.name)
            if match:
                val_loss = float(match.group(1))
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_checkpoint_path = chk_path
        
        if best_checkpoint_path is None:
            raise FileNotFoundError(f"No valid checkpoints found in {checkpoint_dir}. Checkpoint names must be like '...valloss_x.xxxx.pt'.")

        logging.info(f"Loading best model from: {best_checkpoint_path} (Validation Loss: {best_loss:.4f})")
        checkpoint = torch.load(best_checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Set the model to evaluation mode. This is crucial!
        # It disables layers like Dropout for consistent inference.
        self.model.eval()

    def translate(self, src_sentence: str):
        """Translates a single English sentence to Chinese using greedy decoding."""
        if not src_sentence.strip():
            return ""

        # Prepare the input
        src_tokens = [self.bos_id] + self.tokenizer.encode(src_sentence).ids + [self.eos_id]
        src = torch.tensor(src_tokens, dtype=torch.long).unsqueeze(0).to(self.device)
        
        # Start decoding
        tgt_tokens = [self.bos_id]
        
        with torch.no_grad(): # Disable gradient calculation for efficiency
            for _ in range(self.config["MAX_SEQ_LEN"]):
                tgt_input = torch.tensor(tgt_tokens, dtype=torch.long).unsqueeze(0).to(self.device)
                
                # Get model predictions
                logits = self.model(src, tgt_input, self.pad_id)
                
                # Get the most likely next token (greedy decoding)
                next_token_id = logits[:, -1, :].argmax(dim=-1).item()
                tgt_tokens.append(next_token_id)
                
                # Stop if the end-of-sentence token is generated
                if next_token_id == self.eos_id:
                    break
        
        # Decode the generated token IDs back to a string
        translated_text = self.tokenizer.decode(tgt_tokens, skip_special_tokens=True)
        return translated_text

def interactive_session():
    """Runs the main interactive translation loop."""
    try:
        translator = Translator(CONFIG)
        translator.load_best_checkpoint()
    except FileNotFoundError as e:
        logging.error(f"Error initializing translator: {e}")
        logging.error("Please make sure you have run the training script and have a valid tokenizer and checkpoint file.")
        return

    print("\n--- ZHEN - 1 Translator ---")
    print("Type an English sentence and press Enter.")
    print("Type 'quit' or 'exit' to close the program.")
    
    while True:
        try:
            source_text = input("\nEnglish > ")
            if source_text.lower() in ['quit', 'exit', 'q']:
                print("Exiting translator. Goodbye!")
                break
            
            if not source_text:
                continue

            translated_text = translator.translate(source_text)
            print(f"Chinese < {translated_text}")

        except KeyboardInterrupt:
            print("\nExiting translator. Goodbye!")
            break
        except Exception as e:
            logging.error(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    interactive_session()
