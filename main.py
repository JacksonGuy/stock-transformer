import numpy as np

from src.model import Model
from src.data import *
from src.globals import *

def MSE(actual, predicted):
    return np.mean(np.square(actual - predicted))

if __name__ == "__main__":
    # Get price information from stocks.
    # Stock list defined in data.py
    df = get_full_history()

    # Sequence data according to INPUT_LENGTH and OUTPUT_LENGTH
    # Use relative sequence function which adds an extra point.
    sequences, sequence_labels = sequence_dataframe_relative(df)

    # Split into separate sets
    train_index = int(len(sequences) * 0.9)
    train_sequences = sequences[:train_index]
    train_labels = sequence_labels[:train_index]
    test_sequences = sequences[train_index:]
    test_labels = sequence_labels[train_index:]

    # Get relative change on sequences
    train_sequences = relative_change(train_sequences)
    train_labels = relative_change(train_labels)
    test_sequences = relative_change(test_sequences)
    test_labels = relative_change(test_labels)

    # Z-Score normalization
    mean, std = np.mean(train_sequences), np.std(train_sequences)
    train_sequences = (train_sequences - mean) / std
    train_labels = (train_labels - mean) / std
    test_sequences = (test_sequences - mean) / std
    test_labels = (test_labels - mean) / std


    # Create model
    model = Model(train_sequences.shape[-1])

    # Train Transformer section
    train_sequences = torch.tensor(train_sequences).float().to(device)
    train_labels = torch.tensor(train_labels).float().to(device)
    print("--- Training Transformer ---")
    model.train_transformer(train_sequences, train_labels, epochs=1)
    print("\n")
    
    # Train Refiner
    print("--- Training Refiner ---")
    model.train_refiner(train_sequences, train_labels)
    print("\n")

    # Test
    test_sequences = torch.tensor(test_sequences).float().to(device)
    predicted = model.predict(test_sequences)
    stats = []
    for x in range(predicted.shape[0]):
        stats.append(MSE(test_labels[x, :, 3], predicted[x, :, 3]))
    mean_mse = np.mean(stats)
    median_mse = np.median(stats)

    print(f"Mean MSE: {mean_mse}")
    print(f"Median MSE: {median_mse}")