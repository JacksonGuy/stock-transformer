import torch
import torch.nn as nn

from src.globals import device
from src.layers import *

class VanillaTransformer(nn.Module):
    def __init__(self, input_size, output_size, feature_length, d_model, num_heads, num_layers, d_ff, dropout):
        super(VanillaTransformer, self).__init__()
        # Use Linear layers instead of Embedding for continuous numerical input
        self.encoder_input_layer = nn.Linear(input_size, d_model)
        self.decoder_input_layer = nn.Linear(input_size, d_model)

        self.positional_encoding = PositionalEncoding(d_model, feature_length)

        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

        self.fc = nn.Linear(d_model, output_size)
        self.dropout = nn.Dropout(dropout)

        self.padding_mask = ()

    def generate_mask(self, size):
        mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
        mask = mask.float().masked_fill(mask == 1, float('-inf'))
        return mask

    def forward(self, src, tgt):
        # Apply positional encoding and dropout
        # (Randomly zeroes elements --> prevents overfitting)
        src_embedded = self.dropout(self.positional_encoding(self.encoder_input_layer(src)))
        tgt_embedded = self.dropout(self.positional_encoding(self.decoder_input_layer(tgt)))

        # Decoder Mask
        tgt_mask = self.generate_mask(tgt.size(1)).to(device)

        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            # Pass None for the mask as it's not used
            enc_output = enc_layer(enc_output, None)

        dec_output = tgt_embedded
        for dec_layer in self.decoder_layers:
            # Pass None for embed mask for now
            dec_output = dec_layer(dec_output, enc_output, None, tgt_mask)

        output = self.fc(dec_output)
        return output
    
class PatchTransformer(nn.Module):
    def __init__(self, input_size, output_size, feature_length, d_model, num_heads, num_layers, d_ff, dropout):
        super(PatchTransformer, self).__init__()

        self.embedding = nn.Linear(input_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, input_size)

        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

        self.fc = nn.Linear(d_model, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # Apply positional encoding and dropout
        # (Randomly zeroes elements --> prevents overfitting)
        src = src.permute(0, 2, 1)
        src_embedded = self.dropout(self.positional_encoding(self.embedding(src)))

        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, None)

        output = self.fc(enc_output)
        return output.permute(0, 2, 1)
    
class StockGPT(nn.Module):
    def __init__(self, input_size, output_size, feature_length, d_model, num_heads, num_layers, d_ff, dropout):
        super(StockGPT, self).__init__()

        self.embedding = nn.Linear(feature_length, d_model)
        self.positional_encoding = PositionalEncoding(d_model, input_size)
        self.decoder_layers = nn.ModuleList([DecoderOnlyLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

        self.fc = nn.Linear(d_model, feature_length)
        self.dropout = nn.Dropout(dropout)

    def generate_mask(self, size):
        mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
        mask = mask.float().masked_fill(mask == 1, float('-inf'))
        return mask

    def forward(self, tgt):
        tgt_embedded = self.dropout(self.positional_encoding(self.embedding(tgt)))

        # Decoder Mask
        tgt_mask = self.generate_mask(tgt.size(1)).to(device)

        dec_output = tgt_embedded
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, tgt_mask)

        output = self.fc(dec_output)
        return output