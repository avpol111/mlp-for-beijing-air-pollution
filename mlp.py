import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('Beijing_PM.csv')
dataset = df[["pm2.5"]]
dataset.fillna(0, inplace=True)
dataset = dataset[24:]
timeseries = dataset.values.astype('float32')

scaler = MinMaxScaler(feature_range=(0, 1))
timeseries = scaler.fit_transform(timeseries.reshape(-1, 1)).reshape(-1)

train_size = int(len(timeseries) * 0.67)
test_size = len(timeseries) - train_size
train, test = timeseries[:train_size], timeseries[train_size:]

def split_sequence(sequence, n_steps):
    X, y = [], []
    for i in range(len(sequence)):
        end_ix = i + n_steps
        if end_ix > len(sequence)-1:
            break
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return torch.tensor(X), torch.tensor(y)

n_steps = 5
X_train, y_train = split_sequence(train, n_steps=n_steps)
X_test, y_test = split_sequence(test, n_steps=n_steps)

model = nn.Sequential(
    nn.Linear(n_steps, 30),
    nn.ReLU(),
    nn.Linear(30, 1)
)

optimizer = optim.Adam(model.parameters())
loss_fn = nn.MSELoss()
loader = data.DataLoader(data.TensorDataset(X_train, y_train), shuffle=True, batch_size=8)

n_epochs = 100
for epoch in range(n_epochs):
    model.train()
    for X_batch, y_batch in loader:
        y_pred = model(X_batch)
        y_batch = y_batch.unsqueeze(1)
        loss = loss_fn(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if epoch % 10 != 0:
        continue
    model.eval()
    with torch.no_grad():
        y_pred = model(X_train)
        y_pred = torch.from_numpy(scaler.inverse_transform(y_pred.reshape(-1,1)))
        y_train_eval = torch.from_numpy(scaler.inverse_transform(y_train.reshape(-1,1)))
        train_rmse = np.sqrt(loss_fn(y_pred, y_train_eval))
        y_pred = model(X_test)
        y_pred = torch.from_numpy(scaler.inverse_transform(y_pred.reshape(-1,1)))
        y_test_eval = torch.from_numpy(scaler.inverse_transform(y_test.reshape(-1,1)))
        test_rmse = np.sqrt(loss_fn(y_pred, y_test_eval))
    print("Epoch %d: train RMSE %.4f, test RMSE %.4f" % (epoch, train_rmse, test_rmse))