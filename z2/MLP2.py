import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
import pandas as pd
import numpy as np
import numpy.typing as npt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def extract_dresden():
    # load general data
    data = pd.read_csv("df.csv")

    # filter only data for Dresden and required for the task
    # required_columns = ['DATE', 'DRESDEN_temp_mean', 'DRESDEN_humidity', 'DRESDEN_wind_speed']
    required_columns = ["DRESDEN_temp_mean", "DRESDEN_humidity", "DRESDEN_wind_speed"]
    dresden_data = data[required_columns]
    print(dresden_data.head())
    print(f"Dataset size: {len(data)}")

    # save Dresden to the separate file
    dresden_data.to_csv("dresden_filtered_weather_data.csv", index=False)


# neural network
class WeatherPredictionNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = nn.Linear(2, 10)
        self.hidden1 = nn.Linear(10, 20)
        self.hidden2 = nn.Linear(20, 10)
        self.output = nn.Linear(10, 1)

    def forward(self, x):
        x = F.relu(self.input(x))
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = self.output(x)  # Remove ReLU for output layer if regression
        return x


# standardize data
def standardize(data: npt.NDArray):
    mean: np.float64 = data.mean(axis=0)
    std: np.float64 = data.std(axis=0)
    return (data - mean) / std, (mean, std)


# unstandardize data
def unstandardize(data: npt.NDArray, mean: np.float64, std: np.float64):
    return data * std + mean


if __name__ == "__main__":
    extract_dresden()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load Dresden dataset
    df = pd.read_csv("dresden_filtered_weather_data.csv")
    df = df[:-1]
    data_numpy = np.array(df)

    # standardizing
    stand_temp, temp_mean_std = standardize(data_numpy[:, 0])  # temperature
    stand_hum, hum_mean_std = standardize(data_numpy[:, 1])  # humidity
    stand_wind, wind_mean_std = standardize(data_numpy[:, 2])  # wind speed
    standartized_data = np.column_stack([stand_temp, stand_hum, stand_wind])

    # convert to tensors
    data_tensor = torch.tensor(standartized_data, dtype=torch.float32)

    # create TensorDataset
    dataset = TensorDataset(data_tensor)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # create DataLoader
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size, shuffle=False)

    # TRAINING
    IS_TRAIN = True
    if IS_TRAIN:
        # initialize the model, loss function, and optimizer
        model = WeatherPredictionNetwork()
        model.to(device)
        criterion = nn.L1Loss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        epochs = 100
        model.train()  # Set the model to training mode
        for epoch in range(epochs):
            for batch_i, data in enumerate(train_loader, 0):
                data = data[0]
                inputs, true_temp = data[:, 1:3].to(device), data[:, 0].to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, true_temp)
                loss.backward()
                optimizer.step()

                if batch_i % 10 == 9:
                    print(
                        f"[Epoch {epoch + 1} -> Batch {batch_i + 1}] -> Loss: {loss.item():.4f}"
                    )

        print("Training Finished")
        # save the model state dict
        torch.save(model.state_dict(), "dresden_weather_prediction_model.pth")

    # EVALUATION
    IS_EVALUATION = True
    if IS_EVALUATION:
        model_eval = WeatherPredictionNetwork()
        model_eval.load_state_dict(torch.load("dresden_weather_prediction_model.pth"))
        model.eval()
        with torch.no_grad():
            test_pred = model(test_x)
            test_loss = loss_fn(test_pred, test_y.unsqueeze(1))
            print(f"Test Loss: {test_loss.item():.4f}")

            # Additional evaluation metrics
            test_pred_np = test_pred.cpu().numpy()  # Convert to NumPy array for sklearn
            test_y_np = test_y.cpu().numpy()  # Convert to NumPy array for sklearn
            mse = mean_squared_error(test_y_np, test_pred_np)
            mae = mean_absolute_error(test_y_np, test_pred_np)
            r2 = r2_score(test_y_np, test_pred_np)

        print(f"Model Evaluation: MSE={mse:.4f}, MAE={mae:.4f}, R2={r2:.4f}")
        print(f"20 first predictions: {all_predictions[:20]}")
        print(f"20 first actuals: {all_actuals[:20]}")
