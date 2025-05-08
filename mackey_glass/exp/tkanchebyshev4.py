'''
Tkann model with four chebyshev layers.
Activation function tanh 
'''

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import tkan.tkan.nn as tnn 
from tkan.tkan.plotting import plot_kan
import matplotlib.pyplot as plt
import numpy as np
from utils import MackeyGlass, create_time_series_dataset, plot_predictions
torch.random.manual_seed(0)

class TKANModel(nn.Sequential):
    def __init__(self, input_size, output_size, hp):
       super(TKANModel, self).__init__()
       self.tl1 = tnn.ChebyshevKan(input_size, hp)  # tkan layer 1
       self.tl2 = tnn.ChebyshevKan(hp, hp)
       self.tl3 = tnn.ChebyshevKan(hp, hp)  # tkan layer 3
       self.tlfinal = tnn.ChebyshevKan(hp, output_size)  # final tkan layer 

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.tanh(self.tl1(x))
        x = F.tanh(self.tl2(x))
        x = F.tanh(self.tl3(x))
        out = self.tlfinal(x)
        return out

def train_model(model, dataloader, criterion, optimizer, epochs=10):
    model.train()
    loss_values = []
    for epoch in range(epochs):
        total_loss = 0.0
        for inputs, targets in dataloader:
            inputs = inputs.float()
            targets = targets.float()

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        loss_values.append(total_loss)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")
    return loss_values

def evaluate_model(model, dataloader):
    model.eval()
    predictions = []
    true_vals = []
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.float()
            outputs = model(inputs)
            predictions.append(outputs.item())
            true_vals.append(targets.item())
    return predictions, true_vals

if __name__ == "__main__":
    # Parameters
    tau = 17.0
    constant_past = 1.2
    lookback_window = 10
    forecasting_horizon = 5
    num_bins = 50
    test_size = 0.2
    epochs = 50
    learning_rate = 0.001

    # Dataset
    dataset = MackeyGlass(
        tau=tau,
        constant_past=constant_past,
        nmg=10,
        beta=0.2,
        gamma=0.1,
        dt=1.0,
        splits=(1000.0, 500.0),
        start_offset=0.0,
        seed_id=0
    )

    data = [dataset[i] for i in range(len(dataset) - forecasting_horizon)]

    train_loader, test_loader, original_data_test, y_test = create_time_series_dataset(
        data,
        lookback_window=lookback_window,
        forecasting_horizon=forecasting_horizon,
        num_bins=num_bins,
        test_size=test_size,
        MSE=True
    )

    # Model setup
    model = TKANModel(input_size=lookback_window, output_size=1, hp=3)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train and evaluate
    loss_values = train_model(model, train_loader, criterion, optimizer, epochs=epochs)
    predictions, true_vals = evaluate_model(model, test_loader)

    # Visualization
    original_data_train = np.array([point[1] for point in train_loader.dataset])
    y_train = np.array([point[1] for point in train_loader.dataset])

    plot_predictions(
        predictions, true_vals,
        predictions, true_vals,
        predictions, true_vals,
        original_data_train, y_train
    )

    # plot the loss
    plt.plot(loss_values)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss over Epochs")
    plt.show()


    # plot kan model
    fig = plt.figure(figsize=(6, 10))
    plot_kan(model)
    plt.show()
