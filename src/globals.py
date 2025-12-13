import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

np.random.seed(42)

# 1 trading day = 6.5 hours
INPUT_LENGTH = 91
OUTPUT_LENGTH = 26

# Vanilla, Patch, StockGPT
transformer_type = "Patch"