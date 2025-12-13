import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from src.globals import *
from src.transformers import VanillaTransformer, PatchTransformer, StockGPT
from src.refiner import Refiner
from src.training import *

class Model:
    def __init__(self, feature_size):
        # Input Parameters
        input_size = INPUT_LENGTH
        output_size = OUTPUT_LENGTH

        # Parameters
        d_model = 512
        num_heads = 8
        num_layers = 8
        d_ff = 2048
        dropout = 0.1

        if transformer_type == "Vanilla":
            self.transformer = VanillaTransformer(
                input_size, output_size, feature_size,
                d_model, num_heads, num_layers, d_ff, dropout
            ).to(device)

        elif transformer_type == "Patch":
            self.transformer = PatchTransformer(
                input_size, output_size, feature_size,
                d_model, num_heads, num_layers, d_ff, dropout
            ).to(device)

        elif transformer_type == "StockGPT":
            self.transformer = StockGPT(
                input_size, output_size, feature_size,
                d_model, num_heads, num_layers, d_ff, dropout
            ).to(device)

        self.spike_refiner = Refiner(feature_size, feature_size)
        self.general_refiner = Refiner(feature_size, feature_size)

    def train_transformer(self, src, tgt, epochs=100):
        train_transformer(self.transformer, src, tgt, epochs=epochs)

    def train_spike_refiner(self, src, tgt, epochs=50):
        train_spike_refiner(self.spike_refiner, self.transformer, src, tgt, epochs=epochs)

    def train_gen_refiner(self, src, tgt, epochs=50):
        train_general_refiner(self.general_refiner, self.transformer, src, tgt, epochs=epochs)

    def predict(self, src):
        if transformer_type == "Vanilla":
            prediction = autoregressive_prediction(self.transformer, src, self.refiner)
        elif transformer_type == "Patch":
            prediction = non_autoregressive_prediction(self.transformer, src, self.refiner)
        elif transformer_type == "StockGPT":
            prediction = autoregressive_prediction(self.transformer, src, self.refiner)

        return prediction