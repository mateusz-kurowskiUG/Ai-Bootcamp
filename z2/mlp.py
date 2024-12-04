from torch import nn
import torch
import torch.nn.functional as F
import polars as pl
from rich import print
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import torch.optim as optim
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = nn.Linear(2, 200)
        self.hidden1 = nn.Linear(200, 400)
        self.hidden2 = nn.Linear(400, 200)
        self.output = nn.Linear(200, 1)

    def forward(self, x):
        x = F.relu(self.input(x))
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = self.output(x)
        return x


useful_cols = ["DE_BILT_temp_mean", "DE_BILT_humidity", "DE_BILT_wind_speed"]


def preprocess():
    df = pl.read_csv("df.csv")[useful_cols]
    df.write_csv("df_preproc.csv")


if __name__ == "__main__":
    preprocess()
    df = pl.read_csv("df_preproc.csv")
    df_matrix = df.to_numpy()
    scaler = StandardScaler()
    scaled_mtx = scaler.fit_transform(df)

    # Convert DataFrame to torch tensor and move to device
    tensor_data = torch.tensor(scaled_mtx, dtype=torch.float32).to(device)

    # Initialize and move model to device
    model = Model().to(device)

    x, y = (
        tensor_data[:, 1:],
        tensor_data[:, 0],
    )

    # Split data into training and testing sets
    train_x, test_x, train_y, test_y = train_test_split(
        x, y, test_size=0.25, random_state=42, shuffle=True
    )

    n_epochs = 100
    batch_size = 32
    batches_per_epoch = len(train_x) // batch_size

    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(n_epochs):
        model.train()
        for i in range(batches_per_epoch):
            start = i * batch_size
            end = start + batch_size

            x_batch = train_x[start:end]
            y_batch = train_y[start:end]
            optimizer.zero_grad()
            y_pred = model(x_batch)
            loss = loss_fn(
                y_pred, y_batch.unsqueeze(1)
            )  # Ensure y_batch is of shape [batch_size, 1]
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{n_epochs}], Loss: {loss.item():.4f}")

    # Evaluate the model
    model.eval()
    with torch.no_grad():
        test_pred = model(test_x)
        test_loss = loss_fn(test_pred, test_y.unsqueeze(1))
        print(f"Test Loss: {test_loss.item():.4f}")

        # Additional evaluation metrics
        test_pred_np = test_pred.cpu().numpy()  # Convert to NumPy array for sklearn
        test_y_np = test_y.cpu().numpy()  # Convert to NumPy array for sklearn

        # unstand_test_pred = scaler.inverse_transform(test_pred)
        # unstand_test_y = scaler.inverse_transform(test_y_np)

        mse = mean_squared_error(test_y_np, test_pred_np)
        mae = mean_absolute_error(test_y_np, test_pred_np)
        r2 = r2_score(test_y_np, test_pred_np)
        # mse = mean_squared_error(unstand_test_y, unstand_test_pred)
        # mae = mean_absolute_error(unstand_test_y, unstand_test_pred)
        # r2 = r2_score(unstand_test_y, unstand_test_pred)

        print(f"Mean Squared Error (MSE): {mse:.4f}")
        print(f"Mean Absolute Error (MAE): {mae:.4f}")
        print(f"R² Score: {r2:.4f}")
