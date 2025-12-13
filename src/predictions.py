import torch
import torch.nn as nn

from src.globals import OUTPUT_LENGTH, transformer_type

def non_autoregressive_prediction(transformer, input_data, refiner=None):
    transformer.eval()

    if refiner:
        refiner.eval()

    with torch.inference_mode():
        out = transformer(input_data)
        
        if refiner:
            residual = refiner(out)
            out += residual
        
        return out.cpu().numpy()

def autoregressive_prediction(transformer, input_data, refiner=None):
    transformer.eval()

    if refiner:
        refiner.eval()

    device = next(transformer.parameters()).device
    input_data = input_data.to(device)

    with torch.inference_mode():
        tgt = input_data[:, -1:, :].clone()

        preds = []
        for _ in range(OUTPUT_LENGTH):
            if transformer_type == "Vanilla":
                out = transformer(input_data, tgt)
            else:
                out = transformer(tgt)

            next_step = out[:, -1:, :]
            preds.append(next_step)
            tgt = torch.cat([tgt, next_step], dim=1)

        preds = torch.cat(preds, dim=1)

        if refiner:
            residual = refiner(preds)
            preds += residual

        return preds.cpu().numpy()