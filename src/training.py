import math
import numpy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from src.globals import device, transformer_type
from src.predictions import *

def train_transformer(model, src_data, tgt_data, epochs=100):
    criterion = nn.SmoothL1Loss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    train_loader = DataLoader(TensorDataset(src_data, tgt_data), batch_size=64, shuffle=False)

    model.train()
    for epoch in range(epochs):
        for x, y in train_loader:
            optimizer.zero_grad()

            x = x.to(device).float()
            y = y.to(device).float()

            if transformer_type == "Patch":
                # PatchTransformer
                output = model(x)
                loss = criterion(output, y)

            elif transformer_type == "Vanilla":
                # Vanilla Transformer
                output = model(x, y[:, :-1, :])
                loss = criterion(output, y[:, 1:, :])

            elif transformer_type == "StockGPT":
                # StockGPT
                output = model(y[:, :-1, :])
                loss = criterion(output, y[:, 1:, :])

            loss.backward()
            optimizer.step()

        print(f"Epoch: {epoch+1}, Loss: {loss.item()}")

def train_spike_refiner(model, transformer, src_data, tgt_data, epochs=50):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    train_loader = DataLoader(TensorDataset(src_data, tgt_data), batch_size=64, shuffle=False)

    transformer.eval()
    model.train()
    for epoch in range(epochs):
        for x, y in train_loader:
            optimizer.zero_grad()

            x = x.to(device).float()
            y = y.to(device).float()

            if transformer_type == "Patch":
                # PatchTransformer
                predicted = non_autoregressive_prediction(transformer, x)
            elif transformer_type == "Vanilla":
                # VanillaTransformer
                predicted = autoregressive_prediction(transformer, x)
            elif transformer_type == "StockGPT":
                # StockGPT
                predicted = autoregressive_prediction(transformer, x)

            predicted = torch.tensor(predicted).to(device).float().detach()

            predicted = predicted.permute(0, 2, 1)
            y = y.permute(0, 2, 1)

            residual_target = y - predicted
            residual_pred = model(predicted)

            loss = criterion(residual_pred, residual_target)
            loss.backward()
            optimizer.step()

        print(f"Epoch: {epoch+1}, Loss: {loss.item()}")

def train_general_refiner(model, transformer, src_data, tgt_data, epochs=50):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    train_loader = DataLoader(TensorDataset(src_data, tgt_data), batch_size=64, shuffle=False)

    transformer.eval()
    model.train()
    for epoch in range(epochs):
        for x, y in train_loader:
            optimizer.zero_grad()

            x = x.to(device).float()
            y = y.to(device).float()

            if transformer_type == "Patch":
                # PatchTransformer
                predicted = non_autoregressive_prediction(transformer, x)
            elif transformer_type == "Vanilla":
                # VanillaTransformer
                predicted = autoregressive_prediction(transformer, x)
            elif transformer_type == "StockGPT":
                # StockGPT
                predicted = autoregressive_prediction(transformer, x)

            predicted = torch.tensor(predicted).to(device).float().detach()

            predicted = predicted.permute(0, 2, 1)
            y = y.permute(0, 2, 1)

            predicted = model(predicted)

            loss = criterion(predicted, y)
            loss.backward()
            optimizer.step()

        print(f"Epoch: {epoch+1}, Loss: {loss.item()}")