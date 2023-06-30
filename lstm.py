import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

from sklearn.model_selection import train_test_split
from torchmetrics import MeanAbsolutePercentageError

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dt = np.load('all_avg.npy').astype('float32')
# dt = np.loadtxt('los_speed.csv', delimiter=',').astype('float32')

feature_size = dt.shape[1]
win_size = 14
pre_size = 1
hidden = 512
lstm_layer = 2
lr = 0.00001

n_epochs = 500
batch_size = 32
train_size = 0.8
shuffle = True


def create_timeseries(dataset, window_size, prediction_size):
    X, y = [], []
    for i in range(len(dataset) - window_size):
        feature = dataset[i:i + window_size]
        target = dataset[i + window_size:i + window_size + prediction_size]
        X.append(feature)
        y.append(target)

    return np.array(X), np.array(y)


class WaterModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=feature_size, hidden_size=hidden,
                            num_layers=lstm_layer, batch_first=True)
        self.linear = nn.Linear(hidden, feature_size)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.linear(x[:, -1:, :])
        return x


def train_model(X_train, y_train, X_test, y_test):
    model = WaterModel().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    mse = nn.MSELoss()
    mae = nn.L1Loss()
    mape = MeanAbsolutePercentageError().to(device)

    loader = data.DataLoader(data.TensorDataset(
        X_train, y_train), shuffle=shuffle, batch_size=batch_size)

    for epoch in range(n_epochs):
        model.train()
        for X_batch, y_batch in loader:
            y_pred = model(X_batch)
            loss = mae(y_pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # Validation
        # if epoch % 10 != 0:
        #     continue
        model.eval()
        with torch.no_grad():
            y_pred = model(X_train)
            # print(X_train.shape)
            # print(y_pred.shape)
            # print(y_train.shape)
            train_mse = mse(y_pred, y_train)
            train_mae = mae(y_pred, y_train)
            train_mape = mape(y_pred, y_train)

            y_pred = model(X_test)
            test_mse = mse(y_pred, y_test)
            test_mae = mae(y_pred, y_test)
            test_mape = mape(y_pred, y_test)

            print("Epoch %d: train MSE %.4f, test MSE %.4f" %
                  (epoch, train_mse, test_mse))
            print("Epoch %d: train RMSE %.4f, test RMSE %.4f" %
                  (epoch, np.sqrt(train_mse.cpu()), np.sqrt(test_mse.cpu())))
            print("Epoch %d: train MAE %.4f, test MAE %.4f" %
                  (epoch, train_mae, test_mae))
            print("Epoch %d: train MAPE %.4f, test MAPE %.4f" %
                  (epoch, train_mape, test_mape))
            print(" ")


if __name__ == '__main__':
    f = open("lstm.txt", "w")

    if shuffle:
        timeseries_X, timeseries_y = create_timeseries(
            dt, window_size=win_size, prediction_size=pre_size)
        X_train, X_test, y_train, y_test = train_test_split(
            timeseries_X, timeseries_y, train_size=train_size, shuffle=True)
    else:
        train_len = int(len(dt) * 0.8)
        train, test = dt[:train_len], dt[train_len:]
        timeseries_X, timeseries_y = create_timeseries(
            dt, window_size=win_size, prediction_size=pre_size)
        X_train, y_train = create_timeseries(
            train, window_size=win_size, prediction_size=pre_size)
        X_test, y_test = create_timeseries(
            test, window_size=win_size, prediction_size=pre_size)

    train_model(torch.tensor(X_train).to(device), torch.tensor(y_train).to(
        device), torch.tensor(X_test).to(device), torch.tensor(y_test).to(device))

    f.close()
